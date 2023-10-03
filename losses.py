import torch
from torch import nn
import vren
import torch.nn.functional as F
import math


class DistortionLoss(torch.autograd.Function):
    """
    Distortion loss proposed in Mip-NeRF 360 (https://arxiv.org/pdf/2111.12077.pdf)
    Implementation is based on DVGO-v2 (https://arxiv.org/pdf/2206.05085.pdf)

    Inputs:
        ws: (N) sample point weights
        deltas: (N) considered as intervals
        ts: (N) considered as midpoints
        rays_a: (N_rays, 3) ray_idx, start_idx, N_samples
                meaning each entry corresponds to the @ray_idx th ray,
                whose samples are [start_idx:start_idx+N_samples]

    Outputs:
        loss: (N_rays)
    """
    @staticmethod
    def forward(ctx, ws, deltas, ts, rays_a):
        loss, ws_inclusive_scan, wts_inclusive_scan = \
            vren.distortion_loss_fw(ws, deltas, ts, rays_a)
        ctx.save_for_backward(ws_inclusive_scan, wts_inclusive_scan,
                              ws, deltas, ts, rays_a)
        return loss

    @staticmethod
    def backward(ctx, dL_dloss):
        (ws_inclusive_scan, wts_inclusive_scan,
        ws, deltas, ts, rays_a) = ctx.saved_tensors
        dL_dws = vren.distortion_loss_bw(dL_dloss, ws_inclusive_scan,
                                         wts_inclusive_scan,
                                         ws, deltas, ts, rays_a)
        return dL_dws, None, None, None


class NeRFLoss(nn.Module):
    def __init__(self, lambda_opacity=1e-3, lambda_distortion=1e-3,loss_type='l2'):
        super().__init__()

        self.lambda_opacity = lambda_opacity
        self.lambda_distortion = lambda_distortion
        self.loss_type = loss_type

    def forward(self, results, target, **kwargs):
        d = {}
        l2_loss = (results['rgb']-target['rgb'])**2

        if self.loss_type=='l2':
            d['rgb'] = l2_loss

        if self.loss_type=='kg': # always output log_sigma
            d['rgb'] = l2_loss/torch.exp(results['u_pred'])+0.5*torch.exp(results['u_pred'])

        if self.loss_type=='uc':
            bins = torch.logspace(0,10.,20)
            kg_loss = l2_loss/torch.exp(results['u_pred'])+0.5*torch.exp(results['u_pred'])
            rgb_p, u_p = self.soft_assignment(l2_loss.sqrt(),results['u_pred'],bins)
            kl_loss = []
            for i in range(rgb_p.size(-1)):
                kl_loss.append(F.kl_div(u_p[:,i],rgb_p[:,i], reduction='mean'))
            kl_loss = torch.cat(kl_loss)
            kl_loss = kl_loss[None,:].expend(rgb_p.size(0),-1)
            d['rgb'] = kg_loss+kl_loss

        o = results['opacity']+1e-10
        # encourage opacity to be either 0 or 1 to avoid floater
        d['opacity'] = self.lambda_opacity*(-o*torch.log(o))

        if self.lambda_distortion > 0:
            d['distortion'] = self.lambda_distortion * \
                DistortionLoss.apply(results['ws'], results['deltas'],
                                     results['ts'], results['rays_a'])

        return d

    def soft_assignment(self, gt, est, bin_weights):
        # gt, est: (N_rays, 3)
        # bin_weights: (N_bins) in logspace
        N_rays, N_channels = gt.size
        N_bins = bin_weights.size(0)

        miu = gt.mean(0).expand(N_bins, -1)  # (Nbins,3)
        sigma = gt.std(0).expand(N_bins, -1)  # (Nbins,3)
        gt_hat = miu + bin_weights[:, None].expand(-1, N_channels) * sigma  # (N_bins,3)
        gt_hat = gt_hat.expand(N_rays, -1, -1)  # (N_rays,N_bins,N_channels)

        gt = gt[:, None, :].expand(-1, N_bins, -1)
        gt_dist = 10 * torch.exp(-(gt_hat - gt) ** 2 / 5)  # (N_rays,N_bins,N_channels)
        gt_p = F.softmax(gt_dist, dim=1)  # (N_rays,N_bins,N_channels)
        gt_p = torch.sum(gt_p, dim=0) / N_rays  # # (N_rays,N_channels)

        est = est[:, None, :].expand(-1, N_bins, -1)
        est_dist = 10 * torch.exp(-(gt_hat - est) ** 2 / 5)
        est_p = F.softmax(est_dist, dim=1)
        est_p = torch.sum(est_p, dim=0) / N_rays
        return gt_p, est_p
