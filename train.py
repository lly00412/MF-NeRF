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

        rgb_act = 'None' if self.hparams.use_exposure else 'Sigmoid'
        self.model = NGP(scale=self.hparams.scale, 
                            hparams=hparams,
                            rgb_act=rgb_act)
        G = self.model.grid_size
        self.model.register_buffer('density_grid',
            torch.zeros(self.model.cascades, G**3))
        self.model.register_buffer('grid_coords',
            create_meshgrid3d(G, G, G, False, dtype=torch.int32).reshape(-1, 3))
    def forward(self, batch, split):
        if split=='train':
            poses = self.poses[batch['img_idxs']]
            directions = self.directions[batch['pix_idxs']]
        elif split=='vs':
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

        return render(self.model, rays_o, rays_d, **kwargs)

    def setup(self, stage):
        dataset = dataset_dict[self.hparams.dataset_name]
        kwargs = {'root_dir': self.hparams.root_dir,
                  'downsample': self.hparams.downsample}
        if self.hparams.train_img:
            self.train_dataset = dataset(split=self.hparams.split,
                                         subs=self.hparams.train_img,
                                         seed=self.hparams.fewshot_seed,
                                         **kwargs)
        else:
            self.train_dataset = dataset(split=self.hparams.split,
                                         fewshot=self.hparams.fewshot,
                                         seed=self.hparams.fewshot_seed,
                                         **kwargs)

        if self.hparams.pick_by == 'random':
            train_subs = self.train_dataset.subs
            self.choice = train_subs[-self.hparams.n_view:]
            print(self.choice)

        self.train_dataset.batch_size = self.hparams.batch_size
        self.train_dataset.ray_sampling_strategy = self.hparams.ray_sampling_strategy

        if self.hparams.view_select:
            full_imgs = self.train_dataset.full
            train_subs = self.train_dataset.subs
            train_left = np.delete(np.arange(full_imgs), train_subs)
            # random select 10 more and run the evaluation
            if len(train_left) > 10:
                np.random.seed(self.hparams.fewshot_seed)
                train_left = np.random.choice(train_left,10,replace=False)
            self.test_dataset = dataset(split='train',
                                         subs=train_left,
                                         seed=hparams.fewshot_seed,
                                         **kwargs)

            self.test_dataset.split = 'test'
        else:
            self.test_dataset = dataset(split='test', **kwargs)
            # self.hparams.N_vocab = self.train_dataset.N_vocab + self.test_dataset.N_vocab

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
        if self.hparams.view_select:
            self.hparams.no_save_test = True
        if (not self.hparams.no_save_test) or self.hparams.view_select:
            self.val_dir = f'results/{self.hparams.dataset_name}/{self.hparams.exp_name}'
            os.makedirs(self.val_dir, exist_ok=True)
        if self.hparams.save_output:
            self.out_dir = f'results/{self.hparams.dataset_name}/{self.hparams.exp_name}/output/'
            os.makedirs(self.out_dir, exist_ok=True)
        # for warp
        if self.hparams.warp:
            ref_idx = self.hparams.ref_cam
            self.ref_pose = self.test_dataset.poses[ref_idx]
            self.K = torch.eye(4)
            self.K[:3, :3] = self.test_dataset.K.cpu().clone()

    def render_virtual_cam(self,new_c2w, batch):
        v_batch = {'pose': new_c2w.to(batch['pose']),
                   'img_idxs': batch['img_idxs']}
        v_results = self(v_batch, split='test')
        return v_results

    def render_by_rays(self,ray_samples,batch,ray_batch_size):
        rayloader = DataLoader(ray_samples, num_workers=8, batch_size=ray_batch_size, shuffle=False,
                               pin_memory=True)

        all_results = []
        for ray_ids in rayloader:
            sub_batch = batch
            sub_batch['pix_idxs'] = ray_ids
            sub_results = self(sub_batch, split='vs')
            all_results += [sub_results]
        results = {}
        for k in all_results[0].keys():
            results[k] = torch.cat([r[k].clone() for r in all_results])
        del all_results
        return results
    def validation_step(self, batch, batch_nb):
        # print(batch.keys()) #dict_keys(['pose', 'img_idxs', 'rgb'])
        outputs = {'data':{},
                   'eval':{}}
        rgb_gt = batch['rgb']

        if self.view_select:
            if self.hparams.vs_samples == -1:
                self.hparams.vs_samples = self.test_dataset.img_wh[0] * self.test_dataset.img_wh[1]
            #pix_idxs = np.random.choice(self.test_dataset.img_wh[0] * self.test_dataset.img_wh[1], self.hparams.vs_batch_size, replace=False)
            torch.random.manual_seed(self.hparams.fewshot_seed)
            pix_idxs = torch.randperm(self.test_dataset.img_wh[0] * self.test_dataset.img_wh[1])[:self.hparams.vs_samples]
            results = self.render_by_rays(pix_idxs,batch,self.hparams.vs_batch_size)
            results['pix_idxs'] = pix_idxs
            rgb_gt = rgb_gt[pix_idxs]
        else:
            results = self(batch,split='test')
            results['pix_idxs'] = None

        logs = {}
        logs['img_idxs'] = batch['img_idxs']
        # compute each metric per image
        self.val_psnr(results['rgb'], rgb_gt)
        logs['psnr'] = self.val_psnr.compute()
        self.val_psnr.reset()
        outputs['eval']['psnr'] = logs['psnr'].cpu().numpy()

        w, h = self.test_dataset.img_wh
        if not self.view_select:
            rgb_pred = rearrange(results['rgb'], '(h w) c -> 1 c h w', h=h)
            rgb_gt = rearrange(rgb_gt, '(h w) c -> 1 c h w', h=h)
            self.val_ssim(rgb_pred, rgb_gt)
            logs['ssim'] = self.val_ssim.compute()
            self.val_ssim.reset()
            outputs['eval']['ssim'] = logs['ssim'].cpu().numpy()

            torch.cuda.empty_cache()
            if self.hparams.eval_lpips:
                self.val_lpips(torch.clip(rgb_pred[:, :, :, :w // 2] * 2 - 1, -1, 1),
                               torch.clip(rgb_gt[:, :, :, :w // 2] * 2 - 1, -1, 1))
                score1 = self.val_lpips.compute()
                self.val_lpips.reset()
                self.val_lpips(torch.clip(rgb_pred[:, :, :, w // 2:] * 2 - 1, -1, 1),
                               torch.clip(rgb_gt[:, :, :, w // 2:] * 2 - 1, -1, 1))
                score2 = self.val_lpips.compute()
                self.val_lpips.reset()
                logs['lpips'] = (score1 + score2).mean()
                outputs['eval']['lpips'] = logs['lpips'].cpu().numpy()
        else:
            rgb_gt = rgb_gt[results['pix_idxs']]

        #################################################
        #            render virtual camera
        #################################################
        if self.hparams.render_vcam or self.hparams.pick_by=='warp':
            idx = batch['img_idxs']
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # vargs = {'ref_c2w': batch['pose'].clone().cpu(),
            #          'K': self.test_dataset.K.clone().cpu(),
            #          'device': device,
            #          'ref_depth_map': rearrange(results['depth'].cpu(), '(h w) -> h w', h=h),
            #          'pix_ids': None,
            #          'img_h': self.test_dataset.img_wh[1],
            #          'img_w': self.test_dataset.img_wh[0]}
            vargs = {'ref_c2w': batch['pose'].clone().cpu(),
                     'K': self.test_dataset.K.clone().cpu(),
                     'device': device,
                     'ref_depth_map': results['depth'].cpu(),
                     'dense_map': not self.view_select,
                     'pix_ids': results['pix_idxs']}

            Vcam = GetVirtualCam(vargs)
            thetas = [1,-1, 1, -1]
            rot_ax = ['x','x','y','y']
            warp_depths = [vargs['ref_depth_map']]
            counts = 0
            for theta, ax in zip(thetas,rot_ax):
                new_c2w = Vcam.get_near_c2w(batch['pose'].clone().cpu(), theta=theta, axis=ax)

                if not self.hparams.view_select:
                    warp_func = warp_tgt_to_ref
                else:
                    warp_func = warp_tgt_to_ref_sparse

                ##################################################
                #      render vcams and warp to the center cam
                ###################################################
                # rot_results = self.render_virtual_cam(new_c2w, batch)
                #rot_pred = rearrange(rot_results['rgb'].cpu().numpy(), '(h w) c -> h w c', h=h)
                #rot_pred = (rot_pred * 255).astype(np.uint8)
                #imageio.imsave(os.path.join(self.val_dir, f'{idx:03d}_pred_rot{ax}{theta}.png'), rot_pred)

                # rot_depth = rearrange(rot_results['depth'].cpu(), '(h w) -> h w', h=h)
                # warp_depth = warp_tgt_to_ref(rot_depth, batch['pose'], new_c2w, self.test_dataset.K,device).cpu()  # sorted
                # counts += (warp_depth>0)
                # warp_depths += [warp_depth]

                ##################################################
                #      render center cams and warp to the vcams --- has problems!!!!
                ###################################################
                # warp_depth = warp_func(results['depth'].cpu(), new_c2w, batch['pose'],
                #                     self.test_dataset.K,
                #                        results['pix_idxs'], (h,w), device).cpu()
                # warp_depths += [warp_depth]
                # counts += (warp_depth > 0)

            warp_depths = torch.stack(warp_depths)
            warp_sigmas = warp_depths.std(0)
            warp_u = torch.zeros_like(warp_sigmas)
            warp_u[counts>0] = warp_sigmas[counts>0]
            # imageio.imsave(os.path.join(self.val_dir, f'{idx:03d}_warpu.png'), err2img(warp_u.cpu().numpy()))

            warp_score = torch.median(warp_sigmas[counts>0].flatten())
            logs['warp'] = warp_score.cpu()

        ###################################################
        #              MC-Dropout
        ###################################################

        if self.hparams.mcdropout or self.hparams.pick_by=='mcd':
            enable_dropout(self.model.rgb_net,p=self.hparams.p)
            mcd_rgb_preds = []
            print('Start MC-Dropout...')

            #TODO: E[(x-miu)^2] = E[x^2]-miu^2
            N_passes = self.hparams.n_passes
            for N in trange(N_passes):
                if self.hparams.view_select:
                    mcd_results = self.render_by_rays(results['pix_idxs'],batch,self.hparams.vs_batch_size)
                else:
                    mcd_results = self(batch,split='test')
                mcd_rgb_preds.append(mcd_results['rgb']) # (h w) c
                # mcd += mcd_results['rgb']
                # mcd_squre += mcd_results['rgb'] ** 2
                del mcd_results
            mcd_rgb_preds = torch.stack(mcd_rgb_preds,0) # n (h w) c
            results['mcd'] = mcd_rgb_preds.mean(-1).std(0) # (h w)
            close_dropout(self.model.rgb_net)

            mcd_score = torch.median(results['mcd'].flatten())
            logs['mcd'] = mcd_score.cpu()

        if self.hparams.plot_roc:
            img_id = batch['img_idxs']
            ROC_dict = {}
            AUC_dict = {}
            rgb_pred = rearrange(results['rgb'].cpu(), '(h w) c -> h w c', h=h)
            rgb_gt = rearrange(batch['rgb'].cpu(), '(h w) c -> h w c', h=h)
            rgb_err = (rgb_pred-rgb_gt)**2
            rgb_err = rgb_err.mean(-1).flatten().numpy()
            cam_flag = False
            mask_pt = False

            if self.hparams.render_vcam:
                warp_u = warp_u.cpu().flatten().numpy()
                ROC_dict['warp_u'], AUC_dict['warp_u'] = compute_roc(rgb_err, warp_u)
                mask_pt = True

            if self.hparams.mcdropout:
                mcd = results['mcd'].cpu().numpy()
                if mask_pt:
                    ROC_dict['mcd'], AUC_dict['mcd'] = compute_roc(rgb_err,mcd)
                else:
                    ROC_dict['mcd'], AUC_dict['mcd'] = compute_roc(rgb_err, mcd)

            # plot opt
            if mask_pt:
                ROC_dict['rgb_err'], AUC_dict['rgb_err'] = compute_roc(rgb_err, rgb_err)
            else:
                ROC_dict['rgb_err'], AUC_dict['rgb_err'] = compute_roc(rgb_err, rgb_err)

            logs['ROC'] = ROC_dict.copy()
            logs['AUC'] = AUC_dict.copy()

            fig_name = os.path.join(self.val_dir, f'{img_id:03d}_roc.png')
            plot_roc(ROC_dict, fig_name, is_ref_cam=cam_flag, opt_label='rgb_err')

            auc_log = os.path.join(self.val_dir, f'{img_id:03d}_auc.txt')
            with open(auc_log, 'a') as f:
                if self.hparams.mcdropout:
                    f.write(f'MC-Dropout params: \n')
                    f.write(f'n_passes = {self.hparams.n_passes}\n')
                    f.write(f'drop prob = {self.hparams.p}\n')
                if self.hparams.warp:
                    f.write(f'Warp depth params (from tgt to ref): \n')
                    f.write(f'ref cam =  {self.hparams.ref_cam}\n')
                    f.write(f'tgt cam =  {img_id}\n')
                f.write(f' AUC score: \n')
                for key in AUC_dict.keys():
                    f.write(f' {key} auc =  {AUC_dict[key]* 100.:.4f}\n')
                f.close()

        if not self.hparams.no_save_test: # save test image to disk
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

            ########### save other outputs ##################
            if self.hparams.mcdropout:
                mcd = rearrange(results['mcd'].cpu().numpy(), '(h w) -> h w', h=h)
                outputs['data']['mcd'] = mcd
                imageio.imsave(os.path.join(self.val_dir, f'{idx:03d}_mcd.png'), err2img(mcd))

            ########### save outputs ##################
            if self.hparams.save_output:
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

        if not self.hparams.view_select:
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
            if self.hparams.mcdropout:
                ROCs['mcd']= np.stack([x['ROC']['mcd'] for x in outputs]).mean(0)
                AUCs['mcd'] = np.array([x['AUC']['mcd'] for x in outputs]).mean(0)

            if self.hparams.render_vcam:
                ROCs['warp_u'] = np.stack([x['ROC']['warp_u'] for x in outputs]).mean(0)
                AUCs['warp_u'] = np.array([x['AUC']['warp_u'] for x in outputs]).mean(0)

            fig_name = os.path.join(self.val_dir, f'scene_avg_roc.png')
            plot_roc(ROCs, fig_name, opt_label='rgb_err')

            auc_log = os.path.join(self.val_dir, f'scene_avg_auc.txt')
            with open(auc_log, 'a') as f:
                if self.hparams.mcdropout:
                    f.write(f'MC-Dropout params: \n')
                    f.write(f'n_passes = {self.hparams.n_passes}\n')
                    f.write(f'drop prob = {self.hparams.p}\n')
                if self.hparams.warp:
                    f.write(f'Warp depth params (from tgt to ref): \n')
                    f.write(f'ref cam =  {self.hparams.ref_cam}\n')
                f.write(f' AUC score: \n')
                for key in AUCs.keys():
                    f.write(f' {key} auc =  {AUCs[key] * 100.:.4f}\n')
                f.close()

        if self.hparams.view_select:
            if self.hparams.pick_by == 'random':
                self.choice = np.random.choice(self.test_dataset.subs, self.hparams.n_view, replace=False)
            else:
                scores = torch.cat([x[self.hparams.pick_by].reshape(1) for x in outputs])
                img_idxs = torch.from_numpy(self.test_dataset.subs)
                topks = torch.topk(scores,self.hparams.n_view)
                self.choice = img_idxs[topks.indices].tolist()

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items


if __name__ == '__main__':
    start = time.time()
    hparams = get_opts()

    ori_exp_name = hparams.exp_name

    # view selection must run the validation first except for random choice
    if hparams.view_select:
        print('Starting view selection！')
        hparams.exp_name = os.path.join(ori_exp_name, hparams.pick_by, 'vs')
        hparams.val_only = True
        hparams.no_save_test = True
        if hparams.pick_by == 'random':
            hparams.fewshot = hparams.fewshot+hparams.n_view
            hparams.val_only = False
            hparams.view_select = False
            hparams.retrain = False
            hparams.exp_name = os.path.join(ori_exp_name, hparams.pick_by, 'retrain')
            hparams.no_save_test = False

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
                      check_val_every_n_epoch=hparams.num_epochs,
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
    if hparams.view_select:
        view_choices = system.choice
        select_time = time.time()
        time_cost = time.strftime("%H:%M:%S", time.gmtime(select_time - start))
        print(f'View selection by {hparams.pick_by}:  {view_choices}')
        print('Time for selection process: {}'.format(time_cost))

        view_select_log = os.path.join(system.val_dir, f'view_select.txt')
        with open(view_select_log, 'a') as f:
            f.write(f'View Select by: {hparams.pick_by}\n')
            f.write(f'Selected views: {system.choice}\n')
            f.write(f'Time for selection process: {time_cost}\n')
            f.close()

    if hparams.retrain:
        hparams.val_only = False
        hparams.view_select = False
        hparams.no_save_test = False
        hparams.train_img = view_choices+system.train_dataset.subs.tolist()

        hparams.exp_name = os.path.join(ori_exp_name, hparams.pick_by, 'retrain')

        hparams.pick_by = None

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

        trainer = Trainer(max_epochs=hparams.num_epochs,
                          check_val_every_n_epoch=hparams.num_epochs,
                          callbacks=callbacks,
                          logger=logger,
                          enable_model_summary=False,
                          accelerator='gpu',
                          devices=hparams.num_gpus,
                          strategy=DDPPlugin(find_unused_parameters=False)
                          if hparams.num_gpus > 1 else None,
                          num_sanity_val_steps=-1 if hparams.val_only else 0,
                          precision=16)

        trainer.fit(system)

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


