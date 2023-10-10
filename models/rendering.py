import torch
from .custom_functions import \
    RayAABBIntersector, RayMarcher, VolumeRenderer, VolumeRenderer_with_uncert
from einops import rearrange, reduce, repeat
import vren
import numpy as np
MAX_SAMPLES = 1024
NEAR_DISTANCE = 0.01


@torch.cuda.amp.autocast()
def render(model, rays_o, rays_d, rays_t, **kwargs):
    """
    Render rays by
    1. Compute the intersection of the rays with the scene bounding box
    2. Follow the process in @render_func (different for train/test)

    Inputs:
        model: NGP
        rays_o: (N_rays, 3) ray origins
        rays_d: (N_rays, 3) ray directions
        rays_t: (N_rays), ray time as embedding index

    Outputs:
        result: dictionary containing final rgb and depth
    """
    rays_o = rays_o.contiguous(); rays_d = rays_d.contiguous(); rays_t = rays_t.long()
    _, hits_t, _ = \
        RayAABBIntersector.apply(rays_o, rays_d, model.center, model.half_size, 1)
    hits_t[(hits_t[:, 0, 0]>=0)&(hits_t[:, 0, 0]<NEAR_DISTANCE), 0, 0] = NEAR_DISTANCE

    if kwargs.get('test_time', False):
        render_func = __render_rays_test
    else:
        render_func = __render_rays_train

    results = render_func(model, rays_o, rays_d, hits_t,rays_t, **kwargs)
    for k, v in results.items():
        if kwargs.get('to_cpu', False):
            v = v.cpu()
            if kwargs.get('to_numpy', False):
                v = v.numpy()
        results[k] = v
    return results


@torch.no_grad()
def __render_rays_test(model, rays_o, rays_d, hits_t, rays_t, **kwargs):
    """
    Render rays by

    while (a ray hasn't converged)
        1. Move each ray to its next occupied @N_samples (initially 1) samples 
           and evaluate the properties (sigmas, rgbs) there
        2. Composite the result to output; if a ray has transmittance lower
           than a threshold, mark this ray as converged and stop marching it.
           When more rays are dead, we can increase the number of samples
           of each marching (the variable @N_samples)
    """
    exp_step_factor = kwargs.get('exp_step_factor', 0.)
    results = {}

    if 'a_embedded' not in kwargs.keys():
        a_embedded = model.embedding_a(rays_t)
    else:
        a_embedded = kwargs['a_embedded']
    if model.output_transient:
        if 't_embedded' not in kwargs.keys():
            t_embedded = model.embedding_t(rays_t)
        else:
            t_embedded = kwargs['t_embedded']


    # output tensors to be filled in
    N_rays = len(rays_o)
    device = rays_o.device
    opacity = torch.zeros(N_rays, device=device)
    depth = torch.zeros(N_rays, device=device)
    rgb = torch.zeros(N_rays, 3, device=device)
    if model.output_transient:
        beta = torch.zeros(N_rays, device=device)

    samples = total_samples = 0
    alive_indices = torch.arange(N_rays, device=device)
    # if it's synthetic data, bg is majority so min_samples=1 effectively covers the bg
    # otherwise, 4 is more efficient empirically
    min_samples = 1 if exp_step_factor==0 else 4

    while samples < kwargs.get('max_samples', MAX_SAMPLES):
        N_alive = len(alive_indices)
        if N_alive==0: break

        # the number of samples to add on each ray
        N_samples = max(min(N_rays//N_alive, 64), min_samples)
        samples += N_samples

        xyzs, dirs, deltas, ts, N_eff_samples = \
            vren.raymarching_test(rays_o, rays_d, hits_t[:, 0], alive_indices,
                                  model.density_bitfield, model.cascades,
                                  model.scale, exp_step_factor,
                                  model.grid_size, MAX_SAMPLES, N_samples)

        kwargs['a_embedded'] = a_embedded[alive_indices]
        if model.output_transient:
            kwargs['t_embedded'] = t_embedded[alive_indices]

        total_samples += N_eff_samples.sum()
        xyzs = rearrange(xyzs, 'n1 n2 c -> (n1 n2) c')
        dirs = rearrange(dirs, 'n1 n2 c -> (n1 n2) c')
        valid_mask = ~torch.all(dirs==0, dim=1)
        if valid_mask.sum()==0: break

        N_trunks = 4

        static_sigmas, static_rgbs, \
            transient_sigmas, transient_rgbs, transient_betas \
            = process_trunks(model, N_trunks, valid_mask, xyzs, dirs, device, **kwargs)

        if model.output_transient:
            deltas = rearrange(deltas, 'n1 n2 -> (n1 n2)')
            ts = rearrange(ts, 'n1 n2 -> (n1 n2)')
            opacity[alive_indices], depth[alive_indices], _, rgb[alive_indices], beta[alive_indices] = compute_opacity_and_rgb(model, deltas[alive_indices], ts[alive_indices], static_sigmas[alive_indices], static_rgbs[alive_indices], transient_sigmas[alive_indices], transient_rgbs[alive_indices], transient_betas[alive_indices])
            alive_indices[N_eff_samples==0] = -1
        else:
            static_sigmas = rearrange(static_sigmas, '(n1 n2) -> n1 n2', n2=N_samples)
            static_rgbs = rearrange(static_rgbs, '(n1 n2) c -> n1 n2 c', n2=N_samples)
            vren.composite_test_fw(
                static_sigmas, static_rgbs, deltas, ts,
                hits_t[:, 0], alive_indices, kwargs.get('T_threshold', 1e-4),
                N_eff_samples, opacity, depth, rgb)

        alive_indices = alive_indices[alive_indices>=0] # remove converged rays

    results['opacity'] = opacity
    results['depth'] = depth
    results['rgb'] = rgb
    results['total_samples'] = total_samples # total samples for all rays

    if model.output_transient:
        results['beta'] = beta

    if exp_step_factor==0: # synthetic
        rgb_bg = torch.ones(3, device=device)
    else: # real
        rgb_bg = torch.zeros(3, device=device)
    results['rgb'] += rgb_bg*rearrange(1-opacity, 'n -> n 1')

    return results


import torch


def process_trunks(model, N_trunks, valid_mask, xyzs, dirs, device, **kwargs):

    static_sigmas = torch.zeros(len(xyzs), device=device)
    static_rgbs = []
    transient_sigmas = []
    transient_rgbs = []
    transient_betas = []

    valid_mask_chunks = torch.chunk(valid_mask, chunks=N_trunks)
    xyzs_chunks = torch.chunk(xyzs, chunks=N_trunks)
    dirs_chunks = torch.chunk(dirs, chunks=N_trunks)

    for t_idx in range(N_trunks):
        mask_chunk = valid_mask_chunks[t_idx]
        xyz_chunk = xyzs_chunks[t_idx]
        dirs_chunk = dirs_chunks[t_idx]
        n_rays = torch.zeros(len(xyzs_chunks[t_idx]), device=device)

        static_sigmas_chunk = torch.zeros(n_rays, device=device)
        static_rgbs_chunk = torch.zeros(n_rays, 3, device=device)

        if model.output_transient:
            transient_sigmas_chunk = torch.zeros(n_rays, device=device)
            transient_rgbs_chunk = torch.zeros(n_rays, 3, device=device)
            transient_betas_chunk = torch.zeros(n_rays, device=device)

            static_sigmas_chunk[mask_chunk], _static_rgbs, transient_sigmas_chunk[mask_chunk], _transient_rgbs, _transient_betas = model(xyz_chunk[mask_chunk], dirs_chunk[mask_chunk], **kwargs)
            static_rgbs_chunk[mask_chunk] = _static_rgbs.float()
            transient_rgbs_chunk[mask_chunk] = _transient_rgbs.float()
            transient_betas_chunk[mask_chunk] = _transient_betas.float()

            static_sigmas += [static_sigmas_chunk]
            static_rgbs += [static_rgbs]
            transient_sigmas += [transient_sigmas_chunk]
            transient_rgbs += [transient_rgbs_chunk]
            transient_betas += [transient_betas]

        else:
            static_sigmas_chunk[mask_chunk], _static_rgbs, _, _, _ = model(xyz_chunk[mask_chunk], dirs_chunk[mask_chunk], **kwargs)
            static_rgbs_chunk[mask_chunk] = _static_rgbs.float()
            static_sigmas += [static_sigmas_chunk]
            static_rgbs += [static_rgbs]

    static_sigmas = torch.cat(static_sigmas)
    static_rgbs = torch.cat(static_rgbs)

    if model.output_transient:
        transient_sigmas = torch.cat(transient_sigmas)
        transient_rgbs = torch.cat(transient_rgbs)
        transient_betas = torch.cat(transient_betas)

    return static_sigmas, static_rgbs, transient_sigmas, transient_rgbs, transient_betas

# Usage:
# static_sigmas, static_rgbs, transient_sigmas, transient_rgbs, transient_betas = process_trunks(N_trunks, valid_mask, xyzs, dirs, static_sigmas, static_rgbs, model, **kwargs)


def __render_rays_train(model, rays_o, rays_d, hits_t,rays_t, **kwargs):
    """
    Render rays by
    1. March the rays along their directions, querying @density_bitfield
       to skip empty space, and get the effective sample points (where
       there is object)
    2. Infer the NN at these positions and view directions to get properties
       (currently sigmas and rgbs)
    3. Use volume rendering to combine the result (front to back compositing
       and early stop the ray if its transmittance is below a threshold)
    """
    exp_step_factor = kwargs.get('exp_step_factor', 0.)
    results = {}


    if 'a_embedded' not in kwargs.keys():
        kwargs['a_embedded']= model.embedding_a(rays_t)
    if model.output_transient:
        if 't_embedded' not in kwargs.keys():
            kwargs['t_embedded'] = model.embedding_t(rays_t)

    (rays_a, xyzs, dirs,
    results['deltas'], results['ts'], results['rm_samples']) = \
        RayMarcher.apply(
            rays_o, rays_d, hits_t[:, 0], model.density_bitfield,
            model.cascades, model.scale,
            exp_step_factor, model.grid_size, MAX_SAMPLES)
    #print(rays_a.size()) # N_rays, 3

    for k, v in kwargs.items(): # supply additional inputs, repeated per ray
        if isinstance(v, torch.Tensor):
            kwargs[k] = torch.repeat_interleave(v[rays_a[:, 0]], rays_a[:, 2], 0)
    # sigmas, rgbs, u_preds = model(xyzs, dirs, a_embedded,t_embedded, **kwargs)

    static_sigmas, static_rgbs, transient_sigmas, transient_rgbs, transient_betas = model(xyzs, dirs, **kwargs)

    results['opacity'], results['depth'], results['ws'], results['rgb'], results['beta'] = \
        compute_opacity_and_rgb(model, results['deltas'], results['ts'], static_sigmas, static_rgbs, transient_sigmas, transient_rgbs, transient_betas)

    if model.output_transient:
        results['transient_sigmas'] = transient_sigmas

    return results

def compute_opacity_and_rgb(model, deltas,  ts, static_sigmas, static_rgbs, transient_sigmas, transient_rgbs, transient_betas):
    if model.output_transient:
        static_alphas = 1 - torch.exp(-deltas * static_sigmas)
        transient_alphas = 1 - torch.exp(-deltas * transient_sigmas)
        alphas = 1 - torch.exp(-deltas * (static_sigmas + transient_sigmas))
    else:
        alphas = 1 - torch.exp(-deltas * static_sigmas)

    alphas_shifted = torch.cat([torch.ones_like(alphas[:, :1]), 1 - alphas], -1)  # [1, 1-a1, 1-a2, ...]
    transmittance = torch.cumprod(alphas_shifted[:, :-1], -1)

    if model.output_transient:
        static_ws = static_alphas * transmittance
        transient_ws = transient_alphas * transmittance
    else:
        static_ws = alphas * transmittance

    ws = alphas * transmittance
    opacity = torch.sum(ws, dim=('n1', 'n2'))
    depth = torch.sum(ws * ts, dim=('n1', 'n2'))

    static_rgb = torch.sum(static_ws.unsqueeze(-1) * static_rgbs, dim=('n1', 'n2', 'c'))

    if model.output_transient:
        transient_rgb = torch.sum(transient_ws.unsqueeze(-1) * transient_rgbs, dim=('n1', 'n2', 'c'))
        beta = torch.sum(transient_ws * transient_betas, dim=('n1', 'n2'))
        beta += model.beta_min
    else:
        transient_rgb = torch.zeros_like(transient_rgbs[0])
        beta = torch.tensor(0.0)  # or set to a default value

    rgb = static_rgb + transient_rgb if model.output_transient else static_rgb

    return opacity, depth, ws, rgb, beta