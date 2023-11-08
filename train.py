import torch
from torch import nn
from opt import get_opts
import glob
import imageio
import numpy as np
import cv2
from einops import rearrange
import os

# data
from torch.utils.data import DataLoader
from datasets import dataset_dict
from datasets.ray_utils import axisangle_to_R, get_rays

# models
from kornia.utils.grid import create_meshgrid3d
from models.networks import NGP
from models.rendering import render, MAX_SAMPLES
from models.virtual_cam import GetVirtualCam

# optimizer, losses
from apex.optimizers import FusedAdam
from torch.optim.lr_scheduler import CosineAnnealingLR
from losses import NeRFLoss

# metrics
from torchmetrics import (
    PeakSignalNoiseRatio, 
    StructuralSimilarityIndexMeasure
)
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

# pytorch-lightning
import pytorch_lightning
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.distributed import all_gather_ddp_if_available

from utils import *
# from warp.warp_utils import warp_tgt_to_ref
import time
from tqdm import trange

import warnings; warnings.filterwarnings("ignore")


def err2img(err,flip=False):
    if flip:
        err = 1 - (err / np.quantile(err, 0.9))*0.8
    else:
        err = (err / np.quantile(err, 0.9))*0.8
    err_img = cv2.applyColorMap((err*255).astype(np.uint8),
                                  cv2.COLORMAP_JET)
    return err_img

def depth2img(depth):
    depth = (depth-depth.min())/(depth.max()-depth.min())
    depth_img = cv2.applyColorMap((depth*255).astype(np.uint8),
                                  cv2.COLORMAP_TURBO)
    return depth_img

class NeRFSystem(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.warmup_steps = 256
        self.update_interval = 16

        self.loss = NeRFLoss(lambda_distortion=self.hparams.distortion_loss_w,loss_type=self.hparams.loss)
        self.train_psnr = PeakSignalNoiseRatio(data_range=1)
        self.val_psnr = PeakSignalNoiseRatio(data_range=1)
        self.val_ssim = StructuralSimilarityIndexMeasure(data_range=1)
        if self.hparams.eval_lpips:
            self.val_lpips = LearnedPerceptualImagePatchSimilarity('vgg')
            for p in self.val_lpips.net.parameters():
                p.requires_grad = False

        if self.hparams.loss in ['nll','nllc']:
            self.hparams.uncert = True

        if self.hparams.eval_u:
            if not isinstance(self.hparams.u_by, list):
                self.hparams.u_by = [self.hparams.u_by]

        rgb_act = 'None' if self.hparams.use_exposure else 'Sigmoid'
        self.model = NGP(scale=self.hparams.scale, 
                            hparams=hparams,
                            rgb_act=rgb_act)
        G = self.model.grid_size
        self.model.register_buffer('density_grid',
            torch.zeros(self.model.cascades, G**3))
        self.model.register_buffer('grid_coords',
            create_meshgrid3d(G, G, G, False, dtype=torch.int32).reshape(-1, 3))
        self.star_time = time.time()
        self.vs_time = 0
        if self.hparams.view_select:
            self.vs_log = os.path.join(f"logs/{self.hparams.dataset_name}", self.hparams.exp_name, 'vs_log.txt')

    def forward(self, batch, split, isvs=False):
        if split=='train':
            poses = self.poses[batch['img_idxs']]
            directions = self.directions[batch['pix_idxs']]
        elif (split=='vs' and self.hparams.vs_sample_rate<1):
            poses = batch['pose'].unsqueeze(0).repeat(len(batch['pix_idxs']),1,1)
            directions = self.directions[batch['pix_idxs']]
        else:
            poses = batch['pose']
            directions = self.directions

        if self.hparams.optimize_ext:
            dR = axisangle_to_R(self.dR[batch['img_idxs']])
            poses[..., :3] = dR @ poses[..., :3]
            poses[..., 3] += self.dT[batch['img_idxs']]

        rays_o, rays_d = get_rays(directions, poses)

        kwargs = {'test_time': split =='test',
                  'random_bg': self.hparams.random_bg}
        if self.hparams.scale > 0.5:
            kwargs['exp_step_factor'] = 1/256
        if self.hparams.use_exposure:
            kwargs['exposure'] = batch['exposure']
        if isvs:
            kwargs['to_cpu'] = True

        return render(self.model, rays_o, rays_d, **kwargs)

    def setup(self, stage):
        dataset = dataset_dict[self.hparams.dataset_name]
        kwargs = {'root_dir': self.hparams.root_dir,
                  'downsample': self.hparams.downsample}
        if not self.hparams.view_select:
            self.train_dataset = dataset(split=self.hparams.split,
                                         fewshot=0,
                                         seed=self.hparams.vs_seed,
                                         **kwargs)
        else:
            self.train_dataset = dataset(split=self.hparams.split,
                                             fewshot=self.hparams.start,
                                             seed=self.hparams.vs_seed,
                                             **kwargs)
            with open(self.vs_log, 'a') as f:
                f.write(f'Initial train img: {self.hparams.start}\n')
                f.write(f'Initial train ids: {self.train_dataset.subs}\n')
                f.close()

        self.current_vs = 0

        full_train = self.train_dataset.full
        current_train_list = self.train_dataset.subs
        self.handout_list = np.delete(np.arange(full_train), current_train_list)

        if self.hparams.view_select and self.hparams.vs_by==None:
            while self.current_vs<self.hparams.N_vs:
                np.random.seed(self.hparams.vs_seed)
                choice = np.random.choice(len(self.handout_list), self.hparams.view_step, replace=False)
                current_train_list = np.append(current_train_list, self.handout_list[choice])
                self.handout_list = np.delete(self.handout_list, choice)
                self.current_vs += 1

            self.train_dataset = dataset(split=self.hparams.split,
                                         subs=current_train_list,
                                         seed=self.hparams.vs_seed,
                                         **kwargs)
            with open(self.vs_log, 'a') as f:
                f.write(f'Total train img: {len(current_train_list)}\n')
                f.write(f'Train img ids: {current_train_list}\n')
                f.close()

            # self.hparams.N_vocab = self.train_dataset.N_vocab + self.test_dataset.N_vocab

        self.train_dataset.batch_size = self.hparams.batch_size
        self.train_dataset.ray_sampling_strategy = self.hparams.ray_sampling_strategy

        self.test_dataset = dataset(split='test', **kwargs)

    def configure_optimizers(self):
        # define additional parameters
        self.register_buffer('directions', self.train_dataset.directions.to(self.device))
        self.register_buffer('poses', self.train_dataset.poses.to(self.device))

        if self.hparams.optimize_ext:
            N = len(self.train_dataset.poses)
            self.register_parameter('dR',
                nn.Parameter(torch.zeros(N, 3, device=self.device)))
            self.register_parameter('dT',
                nn.Parameter(torch.zeros(N, 3, device=self.device)))

        load_ckpt(self.model, self.hparams.weight_path)

        net_params = []
        for n, p in self.named_parameters():
            if n not in ['dR', 'dT']: net_params += [p]

        opts = []
        self.net_opt = FusedAdam(net_params, self.hparams.lr, eps=1e-15)
        opts += [self.net_opt]
        if self.hparams.optimize_ext:
            opts += [FusedAdam([self.dR, self.dT], 1e-6)] # learning rate is hard-coded
        net_sch = CosineAnnealingLR(self.net_opt,
                                    self.hparams.num_epochs-1,
                                    self.hparams.lr*0.01)

        return opts, [net_sch]

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          num_workers=16,
                          persistent_workers=True,
                          batch_size=None,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.test_dataset,
                          num_workers=8,
                          batch_size=None,
                          pin_memory=True)

    def ray_dataloader(self, rays, batch_size, n_work,pin):
        return DataLoader(rays,
                          num_workers=n_work,
                          batch_size=batch_size,
                          pin_memory=pin,
                          shuffle=False)

    def viewselect_loader(self):
        return DataLoader(self.handout_dataset,
                          num_workers=8,
                          batch_size=None,
                          pin_memory=True)

    def on_train_start(self):
        self.model.mark_invisible_cells(self.train_dataset.K.to(self.device),
                                        self.poses,
                                        self.train_dataset.img_wh)

    def training_step(self, batch, batch_nb, *args):
        if self.global_step%self.update_interval == 0:
            self.model.update_density_grid(0.01*MAX_SAMPLES/3**0.5,
                                           warmup=self.global_step<self.warmup_steps,
                                           erode=self.hparams.dataset_name=='colmap')

        results = self(batch, split='train')
        loss_d = self.loss(results, batch)
        if self.hparams.use_exposure:
            zero_radiance = torch.zeros(1, 3, device=self.device)
            unit_exposure_rgb = self.model.log_radiance_to_rgb(zero_radiance,
                                    **{'exposure': torch.ones(1, 1, device=self.device)})
            loss_d['unit_exposure'] = \
                0.5*(unit_exposure_rgb-self.train_dataset.unit_exposure_rgb)**2
        loss = sum(lo.mean() for lo in loss_d.values())

        with torch.no_grad():
            self.train_psnr(results['rgb'], batch['rgb'])
        self.log('lr', self.net_opt.param_groups[0]['lr'])
        self.log('train/loss', loss)
        # ray marching samples per ray (occupied space on the ray)
        self.log('train/rm_s', results['rm_samples']/len(batch['rgb']), True)
        # volume rendering samples per ray (stops marching when transmittance drops below 1e-4)
        self.log('train/vr_s', results['vr_samples']/len(batch['rgb']), True)
        self.log('train/psnr', self.train_psnr, True)

        return loss

    def on_validation_start(self):

        torch.cuda.empty_cache()
        if self.current_epoch+1 < self.hparams.num_epochs:
            self.no_save_test = True
            self.save_output = False
        else:
            self.no_save_test = self.hparams.no_save_test
            self.save_output = self.hparams.save_output

        self.val_dir = f'results/{self.hparams.dataset_name}/{self.hparams.exp_name}'
        os.makedirs(self.val_dir, exist_ok=True)

        if self.save_output:
            self.out_dir = f'results/{self.hparams.dataset_name}/{self.hparams.exp_name}/output/'
            os.makedirs(self.out_dir, exist_ok=True)
    def on_validation_epoch_start(self):
        current_time = time.time()
        runtime = current_time - self.star_time
        self.log('train/runtime(mins)', runtime / 60, True)

        if self.hparams.view_select and (self.current_vs<self.hparams.N_vs):
            self.vs_dir = os.path.join(f'results/{self.hparams.dataset_name}/{self.hparams.exp_name}/vs/', f'epoch{self.current_epoch}')
            os.makedirs(self.vs_dir, exist_ok=True)

            if not (self.current_epoch+1) % self.hparams.epoch_step:
                print(f'Starting view selection Round {self.current_vs}!')
                vs_start = time.time()
                if self.hparams.vs_by == 'random':
                    np.random.seed(self.hparams.vs_seed)
                    choice = np.random.choice(len(self.handout_list), self.hparams.view_step, replace=False)
                else:
                    dataset = dataset_dict[self.hparams.dataset_name]
                    kwargs = {'root_dir': self.hparams.root_dir,
                              'downsample': self.hparams.downsample}
                    self.handout_dataset = dataset(split='train',
                                                   subs = self.handout_list,
                                                   seed=self.hparams.vs_seed,
                                                    **kwargs)
                    self.handout_dataset.split = 'test'
                    vs_loader = self.viewselect_loader()
                    view_uncert_scores = []
                    for batch_idx, batch in enumerate(vs_loader):
                        vs_score = self.view_select_step(batch,batch_idx)
                        view_uncert_scores.append(vs_score)

                    scores = torch.cat([x.reshape(1) for x in view_uncert_scores])
                    topks = torch.topk(scores, self.hparams.view_step)
                    choice = topks.indices.numpy()

                new_train_list = np.append(self.train_dataset.subs, self.handout_list[choice])
                vs_choice = self.handout_list[choice]
                self.handout_list = np.delete(self.handout_list, choice)
                dataset = dataset_dict[self.hparams.dataset_name]
                kwargs = {'root_dir': self.hparams.root_dir,
                          'downsample': self.hparams.downsample}
                self.train_dataset = dataset(split=self.hparams.split,
                                                 subs=new_train_list,
                                                 seed=self.hparams.vs_seed,
                                                 **kwargs)
                self.current_vs += 1
                vs_end = time.time()
                self.vs_time += vs_end-vs_start
                self.log('vs/runtime(mins)', self.vs_time/60)

                time_cost = time.strftime("%H:%M:%S", time.gmtime(vs_end-vs_start))
                print(f'View selection by {self.hparams.vs_by}:  {vs_choice}')
                print('Time for selection process: {}'.format(time_cost))

                with open(self.vs_log, 'a') as f:
                    f.write(f'VS Round {self.current_vs}\n')
                    f.write(f'View Select by: {self.hparams.vs_by}\n')
                    f.write(f'Sample rate: {self.hparams.vs_sample_rate}')
                    f.write(f'Selected views: {vs_choice}\n')
                    f.write(f'Time for selection process: {time_cost}\n')
                    f.close()

    def render_virtual_cam(self,new_c2w, batch):
        v_batch = {'pose': new_c2w.to(batch['pose']),
                   'img_idxs': batch['img_idxs']}
        v_results = self(v_batch, split='test')
        return v_results

    def render_by_rays(self,ray_samples,batch,ray_batch_size,n_worker=4,pin_m=True):
        # rayloader = DataLoader(ray_samples, num_workers=n_worker, batch_size=ray_batch_size, shuffle=False,
        #                        pin_memory=pin_m)
        rayloader = self.ray_dataloader(ray_samples,batch_size=ray_batch_size,n_work=n_worker,pin=pin_m)

        all_results = []
        for ray_ids in rayloader:
            sub_batch = batch
            sub_batch['pix_idxs'] = ray_ids
            sub_results = self(sub_batch, split='vs',isvs=True)
            all_results += [sub_results]
        results = {}
        for k in ['rgb','depth']:
            if all_results[0][k].dim() == 0:
                results[k] = torch.cat([r[k].clone().reshape(1) for r in all_results])
            else:
                results[k] = torch.cat([r[k].clone() for r in all_results])
        del all_results
        return results

    def warp_uncert(self, batch, results, img_h, img_w, K, isdense=True):
        opacity = results['opacity'].cpu()  # (n_rays)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vargs = {'ref_c2w': batch['pose'].clone().cpu(),
                 'K': K.clone().cpu(),
                 'device': device,
                 'ref_depth_map': results['depth'].cpu(),
                 'opacity': opacity,
                 'dense_map': isdense,
                 'pix_ids': results['pix_idxs'],
                 'img_h': img_h,
                 'img_w': img_w}

        Vcam = GetVirtualCam(vargs)
        thetas = [self.hparams.theta, -self.hparams.theta,self.hparams.theta , -self.hparams.theta]
        rot_ax = ['x', 'x', 'y', 'y']
        warp_depths = [vargs['ref_depth_map'].cpu()]
        counts = 0
        for theta, ax in zip(thetas, rot_ax):
            new_c2w = Vcam.get_near_c2w(batch['pose'].clone().cpu(), theta=theta, axis=ax)

            if self.hparams.view_select and self.hparams.vs_sample_rate < 1:
                warp_func = warp_tgt_to_ref_sparse
            else:
                warp_func = warp_tgt_to_ref

            warp_depth, out_pix_idxs = warp_func(results['depth'].cpu(), new_c2w, batch['pose'],
                                                 K,
                                                 results['pix_idxs'], (img_h, img_w), device)
            if not isdense:
                warp_depth[out_pix_idxs == 0] = float('nan')
                warp_depth[opacity == 0] = float('nan')
                counts += (out_pix_idxs.cpu() > 0) & (opacity > 0)
            else:
                warp_depth[warp_depth == 0] = float('nan')
                warp_depth[opacity == 0] = float('nan')
                counts += (warp_depth.cpu() > 0) & (opacity > 0)
            warp_depth = warp_depth.cpu()
            warp_depths += [warp_depth]

        warp_depths = torch.stack(warp_depths)
        warp_sigmas = np.nanstd(warp_depths.cpu().numpy(), axis=0)
        warp_sigmas = torch.from_numpy(warp_sigmas)

        warp_score = torch.median(warp_sigmas[counts > 0].flatten())
        return warp_sigmas.cpu(), counts.cpu(), warp_score.cpu()

    def mcd_uncert(self, batch, mcd_val='depth', pix_idxs=None, isdense=True):
        enable_dropout(self.model.rgb_net, p=self.hparams.p)
        mcd_preds = []
        print('Start MC-Dropout...')
        counts = 0
        # TODO: E[(x-miu)^2] = E[x^2]-miu^2
        N_passes = self.hparams.n_passes
        for N in trange(N_passes):
            if not isdense:
                mcd_results = self.render_by_rays(pix_idxs, batch, self.hparams.vs_batch_size)
            else:
                mcd_results = self(batch, split='test')
            opacity = mcd_results['opacity']
            mcd_pred = mcd_results[mcd_val]
            _mcd_pred = torch.full_like(mcd_pred, float("nan"))
            _mcd_pred[opacity > 0] = mcd_pred[opacity > 0]
            counts += (opacity > 0)
            mcd_preds.append(_mcd_pred)  # (h w) c
            # mcd += mcd_results['rgb']
            # mcd_squre += mcd_results['rgb'] ** 2
            del mcd_results
        close_dropout(self.model.rgb_net)
        mcd_preds = torch.stack(mcd_preds, 0)  # rgb: n (h w) c    depth: n (h w)
        if mcd_preds.ndim > 2:
            mcd_preds = mcd_preds.mean(-1)
            mcd_preds = mcd_preds.cpu().numpy()
        mcd_sigmas = np.nanstd(mcd_preds, 0)
        mcd_sigmas = torch.from_numpy(mcd_sigmas)

        mcd_score = torch.median(mcd_sigmas[counts > 0].flatten())
        return  mcd_sigmas.cpu(), counts.cpu(), mcd_score.cpu()

    def view_select_step(self, batch,batch_nb):
        img_id = self.hanout_dataset.subs[batch_nb]
        img_w, img_h = self.handout_dataset.img_wh

        if self.hparams.vs_sample_rate < 1:
            total_rays = self.handout_dataset.img_wh[0] * self.handou_dataset.img_wh[1]
            if self.hparams.vs_sample_rate < 1:
                n_samples = int(self.hparams.vs_sample_rate * total_rays)
                # pix_idxs = np.random.choice(self.test_dataset.img_wh[0] * self.test_dataset.img_wh[1], self.hparams.vs_batch_size, replace=False)
                torch.random.manual_seed(self.hparams.vs_seed)
                pix_idxs = torch.randperm(total_rays)[:n_samples]
                results = self.render_by_rays(pix_idxs, batch, self.hparams.vs_batch_size)
                results['pix_idxs'] = pix_idxs
            else:
                results = self(batch, split='test',isvs=True)
                results['pix_idxs'] = None

            if self.hparams.vs_by == 'warp':
                K = self.handout_dataset.K
                sigmas, counts, u_score = self.warp_uncert(batch, results,
                                                           img_h, img_w, K,
                                                            isdense=not (self.hparams.vs_sample_rate<1))
            if self.hparams.vs_by in ['mcd_d', 'mcd_r']:
                mcd_val = 'depth' if self.hparams.vs_by=='mcd_d' else 'rgb'
                sigmas, counts, u_score = self.mcd_uncert(batch,mcd_val,results['pix_idxs'],
                                                          isdense=not (self.hparams.vs_sample_rate<1))

            opacity = results['opacity'].cpu()
            print(f'img {img_id} uncert score:{u_score}')
            print(f'Total resolution: {img_h * img_w}')
            print(f'count pxs: {(counts > 0).sum()}')
            print(f'valid pxs: {(opacity > 0).sum()}')

            if not self.hparams.no_save_vs:
                sigmas = err2img(sigmas[counts > 0].cpu().numpy())  # (n_rays) 1 3
                sigmas = sigmas.squeeze(1)
                counts = counts.cpu().numpy()
                u_img = np.zeros((img_h * img_w, 3)).astype(np.uint8)
                if self.hparams.vs_sample_rate < 1:
                    u_img[results['pix_idxs'].cpu().numpy()][counts > 0] = sigmas
                    u_img = rearrange(u_img, '(h w) c -> h w c', h=img_h)
                else:
                    u_img[counts > 0] = sigmas
                    u_img = rearrange(u_img, '(h w) c -> h w c', h=img_h)
                imageio.imsave(os.path.join(self.vs_dir, f'{img_id:03d}_vsu.png'), u_img)

            return u_score

    def validation_step(self, batch, batch_nb):
        img_id = batch['img_idxs']
        # print(batch.keys()) #dict_keys(['pose', 'img_idxs', 'rgb'])
        outputs = {'data':{},
                   'eval':{}}
        rgb_gt = batch['rgb']

        results = self(batch,split='test')

        logs = {}
        logs['img_idxs'] = batch['img_idxs']
        # compute each metric per image
        self.val_psnr(results['rgb'], rgb_gt.to(results['rgb']))
        logs['psnr'] = self.val_psnr.compute()
        self.val_psnr.reset()
        outputs['eval']['psnr'] = logs['psnr'].cpu().numpy()

        w, h = self.test_dataset.img_wh

        rgb_pred = rearrange(results['rgb'], '(h w) c -> 1 c h w', h=h)
        rgb_gt = rearrange(rgb_gt, '(h w) c -> 1 c h w', h=h)
        self.val_ssim(rgb_pred, rgb_gt)
        logs['ssim'] = self.val_ssim.compute()
        self.val_ssim.reset()
        outputs['eval']['ssim'] = logs['ssim'].cpu().numpy()

        torch.cuda.empty_cache()
        if self.hparams.eval_lpips:
                # if self.hparams.dataset_name=='colmap':
                #     self.val_lpips(torch.clip(rgb_pred[:, :, :, :w // 2] * 2 - 1, -1, 1),
                #                    torch.clip(rgb_gt[:, :, :, :w // 2] * 2 - 1, -1, 1))
                #     score1 = self.val_lpips.compute()
                #     self.val_lpips.reset()
                #     self.val_lpips(torch.clip(rgb_pred[:, :, :, w // 2:] * 2 - 1, -1, 1),
                #                    torch.clip(rgb_gt[:, :, :, w // 2:] * 2 - 1, -1, 1))
                #     score2 = self.val_lpips.compute()
                #     self.val_lpips.reset()
                #     logs['lpips'] = (score1 + score2).mean()
                # else:
            self.val_lpips(torch.clip(rgb_pred[:, :, :, :] * 2 - 1, -1, 1),
                           torch.clip(rgb_gt[:, :, :, :] * 2 - 1, -1, 1))
            score = self.val_lpips.compute()
            self.val_lpips.reset()
            logs['lpips'] = score.mean()
            outputs['eval']['lpips'] = logs['lpips'].cpu().numpy()

        if self.hparams.eval_u:
            ROC_dict = {}
            AUC_dict = {}
            u_dict = {}
            common_counts = 0
            img_w, img_h = self.test_dataset.img_wh
            for u_method in self.hparams.u_by:
                if u_method == 'warp':
                    K = self.test_dataset.K
                    sigmas, counts, u_score = self.warp_uncert(batch, results,
                                                               img_h, img_w, K,
                                                               isdense=True)
                if u_method in ['mcd_d', 'mcd_r']:
                    mcd_val = 'depth' if u_method == 'mcd_d' else 'rgb'
                    sigmas, counts, u_score = self.mcd_uncert(batch, mcd_val, pix_idxs=None,
                                                              isdense=True)

                logs[u_method] = u_score
                common_counts += counts
                u_dict[u_method] = sigmas

                if not self.no_save_test:
                    sigmas = err2img(sigmas[counts > 0].cpu().numpy())  # (n_rays) 1 3
                    sigmas = sigmas.squeeze(1)
                    counts = counts.cpu().numpy()
                    u_img = np.zeros((img_h * img_w, 3)).astype(np.uint8)
                    u_img[counts > 0] = sigmas
                    u_img = rearrange(u_img, '(h w) c -> h w c', h=img_h)
                    imageio.imsave(os.path.join(self.val_dir, f'{img_id:03d}_{self.hparams.u_by}.png'), u_img)

            if self.hparams.plot_roc:
                val_mask = (common_counts>=len(self.hparams.u_by))
                rgb_pred = rearrange(results['rgb'].cpu(), '(h w) c -> h w c', h=h)
                rgb_gt = rearrange(batch['rgb'].cpu(), '(h w) c -> h w c', h=h)
                rgb_err = (rgb_pred-rgb_gt)**2
                val_err = rgb_err[val_mask]
                rgb_err = val_err.mean(-1).flatten().numpy()

                ROC_dict['rgb_err'], AUC_dict['rgb_err'] = compute_roc(rgb_err, rgb_err)

                for u_method in self.hparams.u_by:
                    sigmas = u_dict[u_method][val_mask].numpy()
                    ROC_dict[u_method], AUC_dict[u_method] = compute_roc(rgb_err, sigmas)

            logs['ROC'] = ROC_dict.copy()
            logs['AUC'] = AUC_dict.copy()

            fig_name = os.path.join(self.val_dir, f'{img_id:03d}_roc.png')
            plot_roc(ROC_dict, fig_name, opt_label='rgb_err')

            auc_log = os.path.join(self.val_dir, f'{img_id:03d}_auc.txt')
            with open(auc_log, 'a') as f:
                for u_method in self.hparams.u_by:
                    if u_method in ['mcd_d', 'mcd_r']:
                        f.write(f'{u_method} params: \n')
                        f.write(f'n_passes = {self.hparams.n_passes}\n')
                        f.write(f'drop prob = {self.hparams.p}\n')
                f.write(f' auc socres: \n')
                for key in AUC_dict.keys():
                    f.write(f' {key} auc =  {AUC_dict[key]* 100.:.4f}\n')
                f.close()

        if not self.no_save_test: # save test image to disk
            idx = batch['img_idxs']
            rgb_pred = rearrange(results['rgb'].cpu().numpy(), '(h w) c -> h w c', h=h)
            outputs['data']['rgb_pred'] = rgb_pred

            ###### add errs ###########
            rgb_gt = rearrange(batch['rgb'].cpu().numpy(), '(h w) c -> h w c', h=h)
            outputs['data']['rgb_gt'] = rgb_gt
            err = (rgb_gt - rgb_pred)**2
            rgb_gt = (rgb_gt * 255).astype(np.uint8)
            ############################

            rgb_pred = (rgb_pred*255).astype(np.uint8)
            depth = rearrange(results['depth'].cpu().numpy(), '(h w) -> h w', h=h)
            outputs['data']['depth'] = depth
            imageio.imsave(os.path.join(self.val_dir, f'{idx:03d}_pred.png'), rgb_pred)
            imageio.imsave(os.path.join(self.val_dir, f'{idx:03d}_d.png'), depth2img(depth))
            imageio.imsave(os.path.join(self.val_dir, f'{idx:03d}_gt.png'), rgb_gt)
            imageio.imsave(os.path.join(self.val_dir, f'{idx:03d}_e.png'), err2img(err.mean(-1)))

            ########### save outputs ##################
            if self.save_output:
                idx = batch['img_idxs']
                out_file = check_file_duplication(os.path.join(self.out_dir, f'{idx:03d}.pth'))
                # out_file = os.path.join(self.out_dir, f'{idx:03d}.pth')
                torch.save(outputs,out_file)

            del rgb_gt,rgb_pred,err,depth,results,outputs,batch

        return logs

    def validation_epoch_end(self, outputs):
        psnrs = torch.stack([x['psnr'] for x in outputs])
        mean_psnr = all_gather_ddp_if_available(psnrs).mean()
        self.log('test/psnr', mean_psnr, True)


        ssims = torch.stack([x['ssim'] for x in outputs])
        mean_ssim = all_gather_ddp_if_available(ssims).mean()
        self.log('test/ssim', mean_ssim)

        if self.hparams.eval_lpips:
            lpipss = torch.stack([x['lpips'] for x in outputs])
            mean_lpips = all_gather_ddp_if_available(lpipss).mean()
            self.log('test/lpips_vgg', mean_lpips)

        if self.hparams.plot_roc:
            ROCs = {}
            AUCs = {}
            ROCs['rgb_err'] = np.stack([x['ROC']['rgb_err'] for x in outputs]).mean(0)
            AUCs['rgb_err'] = np.array([x['AUC']['rgb_err'] for x in outputs]).mean(0)

            for u_method in self.hparams.u_by:
                ROCs[u_method] = np.stack([x['ROC'][u_method] for x in outputs]).mean(0)
                AUCs[u_method] = np.array([x['AUC'][u_method] for x in outputs]).mean(0)

            fig_name = os.path.join(self.val_dir, f'scene_avg_roc.png')
            plot_roc(ROCs, fig_name, opt_label='rgb_err')

            auc_log = os.path.join(self.val_dir, f'scene_avg_auc.txt')
            with open(auc_log, 'a') as f:
                for u_method in self.hparams.u_by:
                    if u_method in ['mcd_d', 'mcd_r']:
                        f.write(f'{u_method} params: \n')
                        f.write(f'n_passes = {self.hparams.n_passes}\n')
                        f.write(f'drop prob = {self.hparams.p}\n')
                f.write(f' auc socres: \n')
                for key in AUCs.keys():
                    f.write(f' {key} auc =  {AUCs[key] * 100.:.4f}\n')
                f.close()

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items


if __name__ == '__main__':
    start = time.time()
    hparams = get_opts()

    if hparams.view_select:
        if not hparams.vs_by==None:
            hparams.exp_name = os.path.join(hparams.exp_name,hparams.vs_by)

    pytorch_lightning.seed_everything(hparams.seed)
    if hparams.val_only and (not hparams.ckpt_path):
        raise ValueError('You need to provide a @ckpt_path for validation!')

    if hparams.val_only:
        system = NeRFSystem.load_from_checkpoint(hparams.ckpt_path, strict=False, hparams=hparams)
    else:
        system = NeRFSystem(hparams)

    ckpt_cb = ModelCheckpoint(dirpath=f'ckpts/{hparams.dataset_name}/{hparams.exp_name}',
                              filename='{epoch:d}',
                              save_weights_only=True,
                              every_n_epochs=hparams.num_epochs,
                              save_on_train_epoch_end=True,
                              save_top_k=-1)
    callbacks = [ckpt_cb, TQDMProgressBar(refresh_rate=1)]

    os.makedirs(os.path.join(f"logs/{hparams.dataset_name}", hparams.exp_name), exist_ok=True)
    logger = TensorBoardLogger(save_dir=f"logs/{hparams.dataset_name}",
                               name=hparams.exp_name,
                               default_hp_metric=False)

    trainer = Trainer(max_epochs=0 if hparams.val_only else hparams.num_epochs,
                      check_val_every_n_epoch=hparams.epoch_step if hparams.view_select else hparams.num_epochs,
                      callbacks=callbacks,
                      logger=logger,
                      enable_model_summary=False,
                      accelerator='gpu',
                      devices=hparams.num_gpus,
                      strategy=DDPPlugin(find_unused_parameters=False)
                               if hparams.num_gpus>1 else None,
                      num_sanity_val_steps=-1 if hparams.val_only else 0,
                      precision=16)

    trainer.fit(system)
    # trainer.fit(system, ckpt_path=hparams.ckpt_path)

    if not hparams.val_only: # save slimmed ckpt for the last epoch
        ckpt_ = \
            slim_ckpt(f'ckpts/{hparams.dataset_name}/{hparams.exp_name}/epoch={hparams.num_epochs-1}.ckpt',
                      save_poses=hparams.optimize_ext)
        torch.save(ckpt_, f'ckpts/{hparams.dataset_name}/{hparams.exp_name}/epoch={hparams.num_epochs-1}_slim.ckpt')

    if (not hparams.no_save_test) and \
       hparams.dataset_name=='nsvf' and \
       'Synthetic' in hparams.root_dir: # save video
        imgs = sorted(glob.glob(os.path.join(system.val_dir, '*.png')))
        imageio.mimsave(os.path.join(system.val_dir, 'rgb.mp4'),
                        [imageio.imread(img) for img in imgs[::2]],
                        fps=30, macro_block_size=1)
        imageio.mimsave(os.path.join(system.val_dir, 'depth.mp4'),
                        [imageio.imread(img) for img in imgs[1::2]],
                        fps=30, macro_block_size=1)

    end = time.time()
    runtime = time.strftime("%H:%M:%S", time.gmtime(end - start))
    print('Total runtime: {}'.format(runtime))


