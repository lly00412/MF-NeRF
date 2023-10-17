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
        else:
            poses = batch['pose']
            directions = self.directions

        if self.hparams.optimize_ext:
            dR = axisangle_to_R(self.dR[batch['img_idxs']])
            poses[..., :3] = dR @ poses[..., :3]
            poses[..., 3] += self.dT[batch['img_idxs']]

        rays_o, rays_d = get_rays(directions, poses)

        kwargs = {'test_time': split!='train',
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
        self.train_dataset = dataset(split=self.hparams.split,
                                     fewshot=self.hparams.fewshot,
                                     seed=self.hparams.fewshot_seed,
                                     **kwargs)
        self.train_dataset.batch_size = self.hparams.batch_size
        self.train_dataset.ray_sampling_strategy = self.hparams.ray_sampling_strategy

        self.test_dataset = dataset(split='test', **kwargs)
        self.hparams.N_vocab = self.train_dataset.N_vocab + self.test_dataset.N_vocab

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
        if not self.hparams.no_save_test:
            self.val_dir = f'results/{self.hparams.dataset_name}/{self.hparams.exp_name}'
            os.makedirs(self.val_dir, exist_ok=True)

    def validation_step(self, batch, batch_nb):
        # print(batch.keys()) #dict_keys(['pose', 'img_idxs', 'rgb'])
        outputs = {'data':{},
                   'eval':{}}
        rgb_gt = batch['rgb']
        results = self(batch,split='test')

        logs = {}
        logs['pose'] = batch['pose']
        logs['raw_pose'] = batch['raw_pose']
        # compute each metric per image
        self.val_psnr(results['rgb'], rgb_gt)
        logs['psnr'] = self.val_psnr.compute()
        self.val_psnr.reset()
        outputs['eval']['psnr'] = logs['psnr'].cpu().numpy()

        w, h = self.train_dataset.img_wh
        rgb_pred = rearrange(results['rgb'], '(h w) c -> 1 c h w', h=h)
        rgb_gt = rearrange(rgb_gt, '(h w) c -> 1 c h w', h=h)
        self.val_ssim(rgb_pred, rgb_gt)
        logs['ssim'] = self.val_ssim.compute()
        self.val_ssim.reset()
        outputs['eval']['ssim'] = logs['ssim'].cpu().numpy()
        torch.cuda.empty_cache()
        if self.hparams.eval_lpips:
            self.val_lpips(torch.clip(rgb_pred[:,:,:,:w//2]*2-1, -1, 1),
                           torch.clip(rgb_gt[:,:,:,:w//2]*2-1, -1, 1))
            score1 = self.val_lpips.compute()
            self.val_lpips.reset()
            self.val_lpips(torch.clip(rgb_pred[:, :, :, w//2 :] * 2 - 1, -1, 1),
                           torch.clip(rgb_gt[:, :, :, w//2 :] * 2 - 1, -1, 1))
            score2 = self.val_lpips.compute()
            self.val_lpips.reset()
            logs['lpips'] = (score1+score2).mean()
            outputs['eval']['lpips'] = logs['lpips'].cpu().numpy()

        ###################################################
        #              MC-Dropout
        ###################################################

        if self.hparams.mcdropout:
            enable_dropout(self.model.rgb_net,p=self.hparams.p)
            mcd_rgb_preds = []
            print('Start MC-Dropout...')

            #TODO: E[(x-miu)^2] = E[x^2]-miu^2
            for N in trange(self.hparams.n_passes):
                mcd_results = self(batch,split='test')
                #mcd_rgb_pred = rearrange(mcd_results['rgb'], '(h w) c -> 1 c h w', h=h) # torch (1,3,h,w)
                mcd_rgb_preds.append(mcd_results['rgb']) # (h w) c
                del mcd_results
            mcd_rgb_preds = torch.stack(mcd_rgb_preds,0) # n (h w) c
            results['mcd'] = mcd_rgb_preds.mean(-1).var(0) # (h w) c
            close_dropout(self.model.rgb_net)
        ###################################################

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
            logs['depth'] = rearrange(results['depth'].cpu(), '(h w) -> h w', h=h)
            outputs['data']['depth'] = depth
            imageio.imsave(os.path.join(self.val_dir, f'{idx:03d}_pred.png'), rgb_pred)
            imageio.imsave(os.path.join(self.val_dir, f'{idx:03d}_d.png'), depth2img(depth))
            imageio.imsave(os.path.join(self.val_dir, f'{idx:03d}_gt.png'), rgb_gt)
            imageio.imsave(os.path.join(self.val_dir, f'{idx:03d}_e.png'), err2img(err.mean(-1)))

            ########### save output_transient ##################
            if self.hparams.mcdropout:
                mcd = rearrange(results['mcd'].cpu().numpy(), '(h w) -> h w', h=h)
                outputs['data']['mcd'] = mcd
                imageio.imsave(os.path.join(self.val_dir, f'{idx:03d}_mcd.png'), err2img(mcd))

            ########### save outputs ##################
        if self.hparams.save_output:
            idx = batch['img_idxs']
            torch.save(outputs,os.path.join(self.val_dir, f'{idx:03d}.pth'))

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

        if self.hparams.warp:

            # scale depth back and use the raw poses to do warp (nerf rendered by centered poses)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            raw_poses = torch.stack([x['raw_pose'] for x in outputs])
            poses = torch.stack([x['pose'] for x in outputs])
            depths = torch.stack([x['depth'] for x in outputs])
            N_views = depths.size(0)
            K = self.test_dataset.K
            scale = self.test_dataset.scale
            depths_scale = torch.stack([x['depth'] for x in outputs])
            depths_scale *= scale
            poses_scale = torch.stack([x['pose'] for x in outputs])
            poses_scale[..., 3] *= scale

            imageio.imsave(os.path.join(self.val_dir, f'{0:03d}_dscale.png'),
                           depth2img(depths_scale[0].cpu().numpy()))

            for img_id in trange(1, N_views): # warp depth 0 to other views
                tgt_depth = depths[0]
                tgt_depth_scale = depths_scale[0]
                tgt_x,tgt_y = (2490,2698)

                print(f'depth 000: cord: ({tgt_x},{tgt_y}), value: {tgt_depth[tgt_x,tgt_y]}, scale value:{tgt_depth_scale[tgt_x,tgt_y]}')
                tgt_depth2 = torch.zeros_like(tgt_depth)
                tgt_depth2[tgt_x,tgt_y] = tgt_depth[tgt_x,tgt_y]
                tgt_depth_scale2 = torch.zeros_like(tgt_depth_scale)
                tgt_depth_scale2[tgt_x,tgt_y] = tgt_depth_scale[tgt_x,tgt_y]

                ref_cams = {'pose':poses[img_id],
                            'K': K}
                tgt_cams = {'pose':poses[0],
                            'K':K}
                ref_cams_scale = {'pose': poses_scale[img_id],
                            'K': K}
                tgt_cams_scale = {'pose': poses_scale[0],
                            'K': K}

                warped_depth = warp_tgt_to_ref(tgt_depth, ref_cams, tgt_cams, device)
                imageio.imsave(os.path.join(self.val_dir, f'{0:03d}_to_{img_id:03d}_warpd.png'),
                               depth2img(warped_depth.cpu().numpy()))

                warped_depth_scale = warp_tgt_to_ref(tgt_depth_scale,ref_cams_scale, tgt_cams_scale, device)
                imageio.imsave(os.path.join(self.val_dir, f'{0:03d}_to_{img_id:03d}_warpd_scale.png'),
                               depth2img(warped_depth_scale.cpu().numpy()))

                warped_depth2 = warp_tgt_to_ref(tgt_depth2, ref_cams, tgt_cams, device)
                warped_depth_scale2 = warp_tgt_to_ref(tgt_depth_scale2, ref_cams_scale, tgt_cams_scale, device)

                print(f'using centered poses:')
                print(f'warp 000 to 001 cord: {torch.nonzero(warped_depth2)}, value: {warped_depth2[warped_depth2 > 0]}')
                print(
                    f'warp(scale) 000 to 001 cord: {torch.nonzero(warped_depth_scale2)}, value: {warped_depth_scale2[warped_depth_scale2 > 0]}')

                ref_cams_raw = {'pose': raw_poses[img_id],
                            'K': K}
                tgt_cams_raw = {'pose': raw_poses[0],
                            'K': K}

                raw_poses_norm = raw_poses
                raw_poses_norm[...,3] /= scale
                ref_cams_raw_norm = {'pose': raw_poses_norm[img_id],
                                'K': K}
                tgt_cams_raw_norm = {'pose': raw_poses_norm[0],
                                'K': K}

                warped_depth_raw_scale = warp_tgt_to_ref(tgt_depth_scale, ref_cams_raw, tgt_cams_raw, device)
                imageio.imsave(os.path.join(self.val_dir, f'{0:03d}_to_{img_id:03d}_warpd_raw_scale.png'),
                               depth2img(warped_depth_raw_scale.cpu().numpy()))

                warped_depth_raw_norm = warp_tgt_to_ref(tgt_depth, ref_cams_raw_norm, tgt_cams_raw_norm, device)
                imageio.imsave(os.path.join(self.val_dir, f'{0:03d}_to_{img_id:03d}_warpd_raw.png'),
                               depth2img(warped_depth_raw_norm.cpu().numpy()))

                warped_depth_raw_norm2 = warp_tgt_to_ref(tgt_depth2, ref_cams_raw_norm, tgt_cams_raw_norm, device)
                warped_depth_scale_raw2 = warp_tgt_to_ref(tgt_depth_scale2, ref_cams_raw, tgt_cams_raw, device)
                print(f'using raw poses:')
                print(
                    f'warp 000 to 001 cord: {torch.nonzero(warped_depth_raw_norm2)}, value: {warped_depth_raw_norm2[warped_depth_raw_norm2 > 0]}')
                print(
                    f'warp(scale) 000 to 001 cord: {torch.nonzero(warped_depth_scale_raw2)}, value: {warped_depth_scale_raw2[warped_depth_scale_raw2 > 0]}')

                valid_mask = (warped_depth > 0)
                warp_err = torch.zeros(warped_depth.size()).to(warped_depth)
                depth_gt = depths[img_id,...].to(warped_depth)
                depth_gt_scale = depths_scale[img_id,...].to(warped_depth_scale)
                gt_x,gt_y = (2439,2583)

                print(f'depth 001: cord: ({gt_x}, {gt_y}), value: {depth_gt[gt_x,gt_y]}, scale value: {depth_gt_scale[gt_x,gt_y]}')

                warp_err[valid_mask] = (depth_gt[valid_mask] - warped_depth[valid_mask]) ** 2

                warp_err = warp_err.cpu().numpy()
                imageio.imsave(os.path.join(self.val_dir, f'{0:03d}_to_{img_id:03d}_warpe.png'), err2img(warp_err))
                imageio.imsave(os.path.join(self.val_dir, f'{img_id:03d}_dscale.png'),
                               depth2img(depths_scale[img_id].cpu().numpy()))




    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items


if __name__ == '__main__':
    start = time.time()
    hparams = get_opts()
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
