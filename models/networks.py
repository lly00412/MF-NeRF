import torch
from torch import nn
import tinycudann as tcnn
import vren
from einops import rearrange
from .custom_functions import TruncExp
import numpy as np

from .rendering import NEAR_DISTANCE


class NGP(nn.Module):
    # def __init__(self, scale, hparams, rgb_act='Sigmoid',uncert=False):
    def __init__(self, scale, hparams, rgb_act='Sigmoid',
                 in_channels_a=48,
                 in_channels_t=16,
                 beta_min=0.03):
        beta_min = 0.03
        super().__init__()
        """
        ---Parameters for NeRF-W (used in fine model only as per section 4.3)---
        ---cf. Figure 3 of the paper---
        encode_appearance: whether to add appearance encoding as input (NeRF-A)
        in_channels_a: appearance embedding dimension. n^(a) in the paper
        encode_transient: whether to add transient encoding as input (NeRF-U)
        in_channels_t: transient embedding dimension. n^(tau) in the paper
        beta_min: minimum pixel color variance
        """



        if not rgb_act=='None':
            self.rgb_act = nn.Sigmoid()

        # scene bounding box
        self.scale = scale
        self.output_transient = hparams.output_transient
        self.register_buffer('center', torch.zeros(1, 3))
        self.register_buffer('xyz_min', -torch.ones(1, 3)*scale)
        self.register_buffer('xyz_max', torch.ones(1, 3)*scale)
        self.register_buffer('half_size', (self.xyz_max-self.xyz_min)/2)

        # each density grid covers [-2^(k-1), 2^(k-1)]^3 for k in [0, C-1]
        self.cascades = max(1+int(np.ceil(np.log2(2*scale))), 1)
        self.grid_size = 128
        self.register_buffer('density_bitfield',
            torch.zeros(self.cascades*self.grid_size**3//8, dtype=torch.uint8))

        # Nerf-W

        self.embedding_a = nn.Embedding(hparams.N_vocab, hparams.N_a)
        self.embedding_t = nn.Embedding(hparams.N_vocab, hparams.N_tau)

        self.encode_appearance = hparams.encode_a
        self.in_channels_a = in_channels_a if hparams.encode_a else 0
        self.encode_transient = hparams.encode_t
        self.in_channels_t = in_channels_t
        self.beta_min = beta_min

        # constants
        L = hparams.L; F = hparams.F; log2_T = hparams.T; N_min = hparams.N_min; N_tables = hparams.N_tables
        b = np.exp(np.log(hparams.N_max*scale/N_min)/(L-1))
        print(f'GridEncoding: Nmin={N_min} b={b:.5f} F={F} T=2^{log2_T} L={L}')

        self.xyz_encoder = \
            tcnn.NetworkWithInputEncoding(
                n_input_dims=3, n_output_dims=16,
                encoding_config={
                    "otype": f"{hparams.grid}Grid",    # HashGrid / WindowGrid
                    "type": hparams.grid,     # Hash / Window
                    "n_levels": L,
                    "n_features_per_level": F,
                    "log2_hashmap_size": log2_T,
                    "base_resolution": N_min,
                    "n_tables": N_tables,
                    "per_level_scale": b,
                    "interpolation": "Linear"
                },
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": 64,
                    "n_hidden_layers": 1,
                }
            )
        print(f'# features = {self.xyz_encoder.params.shape[0]-3072}')

        self.dir_encoder = \
            tcnn.Encoding(
                n_input_dims=3,
                encoding_config={
                    "otype": "SphericalHarmonics",
                    "degree": 4,
                },
            )

        self. dir_a_encoder = nn.Sequential(
                        nn.Linear(16+self.in_channels_a, 16), nn.ReLU(True))

        self.static_rgb_net = nn.Sequential(
            nn.Linear(32, 128,bias=False),
            nn.ReLU(),
            nn.Linear(128, 128,bias=False),
            nn.ReLU(),
            nn.Linear(128, 128,bias=False),
            nn.ReLU(),
            nn.Dropout(p=0),
            nn.Linear(128, 3,bias=False),
            self.rgb_act,
        )

        if self.encode_transient:
            # transient encoding layers
            self.transient_encoding = torch.nn.Sequential(
                nn.Linear(16+in_channels_t, 128), nn.ReLU(True),
                nn.Linear(128, 128), nn.ReLU(True),
                nn.Linear(128, 128), nn.ReLU(True),
                nn.Linear(128,128), nn.ReLU(True))
            # transient output layers
            self.transient_sigma = nn.Sequential(nn.Linear(128, 1), nn.Softplus())
            self.transient_rgb = nn.Sequential(nn.Dropout(p=0),nn.Linear(128, 3), self.rgb_act)
            self.transient_beta = nn.Sequential(nn.Linear(128, 1), nn.Softplus())

        if self.rgb_act == 'None': # rgb_net output is log-radiance
            for i in range(3): # independent tonemappers for r,g,b
                tonemapper_net = \
                    tcnn.Network(
                        n_input_dims=1, n_output_dims=1,
                        network_config={
                            "otype": "FullyFusedMLP",
                            "activation": "ReLU",
                            "output_activation": "Sigmoid",
                            "n_neurons": 64,
                            "n_hidden_layers": 1,
                        }
                    )
                setattr(self, f'tonemapper_net_{i}', tonemapper_net)

    def density(self, x, return_feat=False):
        """
        Inputs:
            x: (N, 3) xyz in [-scale, scale]
            return_feat: whether to return intermediate feature

        Outputs:
            sigmas: (N)
        """
        x = (x-self.xyz_min)/(self.xyz_max-self.xyz_min)
        h = self.xyz_encoder(x)
        sigmas = TruncExp.apply(h[:, 0])
        if return_feat: return sigmas, h
        return sigmas

    def log_radiance_to_rgb(self, log_radiances, **kwargs):
        """
        Convert log-radiance to rgb as the setting in HDR-NeRF.
        Called only when self.rgb_act == 'None' (with exposure)

        Inputs:
            log_radiances: (N, 3)

        Outputs:
            rgbs: (N, 3)
        """
        if 'exposure' in kwargs:
            log_exposure = torch.log(kwargs['exposure'])
        else: # unit exposure by default
            log_exposure = 0

        out = []
        for i in range(3):
            inp = log_radiances[:, i:i+1]+log_exposure
            out += [getattr(self, f'tonemapper_net_{i}')(inp)]
        rgbs = torch.cat(out, 1)
        return rgbs

    def forward(self, x, d, a, t, **kwargs):
        """
        Inputs:
            x: (N, 3) xyz in [-scale, scale]
            d: (N, 3) directions
            a: (N, in_channels_a) apperientszu
            t: (N, in_channels_t) transient

        Outputs:
            sigmas: (N)
            rgbs: (N, 3)

        """

        # static network
        static_sigmas, h = self.density(x, return_feat=True)  # h (N,16)
        d = d/torch.norm(d, dim=1, keepdim=True)
        d = self.dir_encoder((d+1)/2) # (N,16)
        if self.encode_appearance:
            d = self.dir_a_encoder(torch.cat([d,a],1))
        static_rgbs = self.static_rgb_net(torch.cat([d, h], 1))

        # transient network
        if self.output_transient:
            if self.encode_transient:
                transient_encoding_input = torch.cat([t, h], 1)
                t = self.transient_encoding(transient_encoding_input)
            transient_sigmas = self.transient_sigma(t)  # (B, 1)
            transient_rgbs = self.transient_rgb(t)  # (B, 3)
            transient_betas = self.transient_beta(t)  # (B, 1)
        else:
            transient_sigmas = None
            transient_rgbs = None
            transient_betas = None

        if self.rgb_act == 'None':  # rgbs is log-radiance
            if kwargs.get('output_radiance', False):  # output HDR map
                static_rgbs = TruncExp.apply(static_rgbs)
                if self.output_transient:
                    transient_rgbs = TruncExp.apply(transient_rgbs)
            else:  # convert to LDR using tonemapper networks
                static_rgbs = self.log_radiance_to_rgb(static_rgbs, **kwargs)
                if self.output_transient:
                    transient_rgbs = self.log_radiance_to_rgb(transient_rgbs, **kwargs)

        return static_sigmas, static_rgbs, transient_sigmas, transient_rgbs, transient_betas


    @torch.no_grad()
    def get_all_cells(self):
        """
        Get all cells from the density grid.
        
        Outputs:
            cells: list (of length self.cascades) of indices and coords
                   selected at each cascade
        """
        indices = vren.morton3D(self.grid_coords).long()
        cells = [(indices, self.grid_coords)] * self.cascades

        return cells

    @torch.no_grad()
    def sample_uniform_and_occupied_cells(self, M, density_threshold):
        """
        Sample both M uniform and occupied cells (per cascade)
        occupied cells are sample from cells with density > @density_threshold
        
        Outputs:
            cells: list (of length self.cascades) of indices and coords
                   selected at each cascade
        """
        cells = []
        for c in range(self.cascades):
            # uniform cells
            coords1 = torch.randint(self.grid_size, (M, 3), dtype=torch.int32,
                                    device=self.density_grid.device)
            indices1 = vren.morton3D(coords1).long()
            # occupied cells
            indices2 = torch.nonzero(self.density_grid[c]>density_threshold)[:, 0]
            if len(indices2)>0:
                rand_idx = torch.randint(len(indices2), (M,),
                                         device=self.density_grid.device)
                indices2 = indices2[rand_idx]
            coords2 = vren.morton3D_invert(indices2.int())
            # concatenate
            cells += [(torch.cat([indices1, indices2]), torch.cat([coords1, coords2]))]

        return cells

    @torch.no_grad()
    def mark_invisible_cells(self, K, poses, img_wh, chunk=64**3):
        """
        mark the cells that aren't covered by the cameras with density -1
        only executed once before training starts

        Inputs:
            K: (3, 3) camera intrinsics
            poses: (N, 3, 4) camera to world poses
            img_wh: image width and height
            chunk: the chunk size to split the cells (to avoid OOM)
        """
        N_cams = poses.shape[0]
        self.count_grid = torch.zeros_like(self.density_grid)
        w2c_R = rearrange(poses[:, :3, :3], 'n a b -> n b a') # (N_cams, 3, 3)
        w2c_T = -w2c_R@poses[:, :3, 3:] # (N_cams, 3, 1)
        cells = self.get_all_cells()
        for c in range(self.cascades):
            indices, coords = cells[c]
            for i in range(0, len(indices), chunk):
                xyzs = coords[i:i+chunk]/(self.grid_size-1)*2-1
                s = min(2**(c-1), self.scale)
                half_grid_size = s/self.grid_size
                xyzs_w = (xyzs*(s-half_grid_size)).T # (3, chunk)
                xyzs_c = w2c_R @ xyzs_w + w2c_T # (N_cams, 3, chunk)
                uvd = K @ xyzs_c # (N_cams, 3, chunk)
                uv = uvd[:, :2]/uvd[:, 2:] # (N_cams, 2, chunk)
                in_image = (uvd[:, 2]>=0)& \
                           (uv[:, 0]>=0)&(uv[:, 0]<img_wh[0])& \
                           (uv[:, 1]>=0)&(uv[:, 1]<img_wh[1])
                covered_by_cam = (uvd[:, 2]>=NEAR_DISTANCE)&in_image # (N_cams, chunk)
                # if the cell is visible by at least one camera
                self.count_grid[c, indices[i:i+chunk]] = \
                    count = covered_by_cam.sum(0)/N_cams

                too_near_to_cam = (uvd[:, 2]<NEAR_DISTANCE)&in_image # (N, chunk)
                # if the cell is too close (in front) to any camera
                too_near_to_any_cam = too_near_to_cam.any(0)
                # a valid cell should be visible by at least one camera and not too close to any camera
                valid_mask = (count>0)&(~too_near_to_any_cam)
                self.density_grid[c, indices[i:i+chunk]] = \
                    torch.where(valid_mask, 0., -1.)

    @torch.no_grad()
    def update_density_grid(self, density_threshold, warmup=False, decay=0.95, erode=False):
        density_grid_tmp = torch.zeros_like(self.density_grid)
        if warmup: # during the first steps
            cells = self.get_all_cells()
        else:
            cells = self.sample_uniform_and_occupied_cells(self.grid_size**3//4,
                                                           density_threshold)
        # infer sigmas
        for c in range(self.cascades):
            indices, coords = cells[c]
            s = min(2**(c-1), self.scale)
            half_grid_size = s/self.grid_size
            xyzs_w = (coords/(self.grid_size-1)*2-1)*(s-half_grid_size)
            # pick random position in the cell by adding noise in [-hgs, hgs]
            xyzs_w += (torch.rand_like(xyzs_w)*2-1) * half_grid_size
            density_grid_tmp[c, indices] = self.density(xyzs_w)

        if erode:
            # My own logic. decay more the cells that are visible to few cameras
            decay = torch.clamp(decay**(1/self.count_grid), 0.1, 0.95)
        self.density_grid = \
            torch.where(self.density_grid<0,
                        self.density_grid,
                        torch.maximum(self.density_grid*decay, density_grid_tmp))

        mean_density = self.density_grid[self.density_grid>0].mean().item()

        vren.packbits(self.density_grid, min(mean_density, density_threshold),
                      self.density_bitfield)
