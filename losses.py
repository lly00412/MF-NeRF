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
    def __init__(self, lambda_opacity=1e-3, lambda_distortion=1e-3,lambda_u=0.01,loss_type='l2'):
        super().__init__()

        self.lambda_opacity = lambda_opacity
        self.lambda_distortion = lambda_distortion
        self.loss_type = loss_type
        self.lambda_u = lambda_u

    def forward(self, results, target, **kwargs):
        d = {}
        l2_loss = (results['rgb']-target['rgb'])**2

        if self.loss_type=='l2':
            d['rgb'] = l2_loss

        if self.loss_type=='nll': # always output log_sigma
            l2_loss = l2_loss.mean(-1)
            d['rgb'] = (l2_loss / (2 * results['beta'] ** 2)) + torch.log(results['beta'])
            # d['t_sigmas'] = self.lambda_u * results['transient_sigmas']

        if self.loss_type=='nllc':
            l2_loss = l2_loss.mean(-1)
            bins = torch.logspace(0,math.log(5),20)
            nll_loss = (l2_loss.mean(-1) / (2 * results['beta'] ** 2)) + torch.log(results['beta'])
            rgb_p, beta_p = self.soft_assignment(l2_loss.sqrt(),results['beta'],bins)
            # kl_loss = []
            # for i in range(rgb_p.size(-1)):
            #     kl_loss.append(F.kl_div(b_p[:,i],rgb_p[:,i], reduction='sum').reshape(1))
            # kl_loss = torch.cat(kl_loss)
            # kl_loss = kl_loss[None,:].expand(l2_loss.size(0),-1)
            kl_loss = F.kl_div(beta_p.log(),rgb_p, reduction='sum')
            d['rgb'] = nll_loss+kl_loss
            # d['t_sigmas'] = self.lambda_u * results['transient_sigmas']

        o = results['opacity']+1e-10
        # # encourage opacity to be either 0 or 1 to avoid floater
        d['opacity'] = self.lambda_opacity*(-o*torch.log(o))
        #
        # d['density'] = (1-d['opacity'])**2

        if self.lambda_distortion > 0:
            d['distortion'] = self.lambda_distortion * \
                DistortionLoss.apply(results['ws'], results['deltas'],
                                     results['ts'], results['rays_a'])

        return d

    # def soft_assignment(self, gt, est, bin_weights):
    #     # gt, est: (N_rays, )
    #     # bin_weights: (N_bins) in logspace
    #     N_rays, N_channels = gt.size()
    #     N_bins = bin_weights.size(0)
    #     bin_weights = bin_weights.to(gt.device)
    #
    #     miu = gt.mean(0).expand(N_bins, -1)  # (Nbins,3)
    #     sigma = gt.std(0).expand(N_bins, -1)  # (Nbins,3)
    #
    #     gt_hat = miu + bin_weights[:, None].expand(-1, N_channels) * sigma  # (N_bins,3)
    #     gt_hat = gt_hat.expand(N_rays, -1, -1)  # (N_rays,N_bins,N_channels)
    #
    #     gt = gt[:, None, :].expand(-1, N_bins, -1)
    #     gt_dist = 10 * torch.exp(-(gt_hat - gt) ** 2 / 5)  # (N_rays,N_bins,N_channels)
    #     gt_p = F.softmax(gt_dist, dim=1)  # (N_rays,N_bins,N_channels)
    #     gt_p = torch.sum(gt_p, dim=0) / N_rays  # # (N_rays,N_channels)
    #
    #     est = est[:, None, :].expand(-1, N_bins, -1)
    #     est_dist = 10 * torch.exp(-(gt_hat - est) ** 2 / 5)
    #     est_p = F.softmax(est_dist, dim=1)
    #     est_p = torch.sum(est_p, dim=0) / N_rays
    #     return gt_p, est_p

    def soft_assignment(self, x, y, bin_weights):
        # gt, est: (N_rays, )
        # bin_weights: (N_bins) in logspace
        bin_weights = bin_weights.to(x.device)

        miu = x.mean()
        sigma = x.std()
        x_hat = miu + bin_weights * sigma
        x_hat = x_hat.repeat(x.size(0), 1)
        x = x[:, None]
        y = y[:, None]
        x_dist = 10 * torch.exp(-(x_hat - x) ** 2 / 5)
        x_p = F.softmax(x_dist, dim=1)
        x_p = torch.sum(x_p, dim=0) / len(x)
        y_dist = 10 * torch.exp(-(x_hat - y) ** 2 / 5)
        y_p = F.softmax(y_dist, dim=1)
        y_p = torch.sum(y_p, dim=0) / len(x)
        return x_p, y_p

