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

def err2img(err):
    err = (err / np.quantile(err, 0.9))*0.8
    err_img = cv2.applyColorMap((err*255).astype(np.uint8),
                                  cv2.COLORMAP_JET)
    return err_img

def depth2img(depth):
    depth = (depth-depth.min())/(depth.max()-depth.min())
    depth_img = cv2.applyColorMap((depth*255).astype(np.uint8),
                                  cv2.COLORMAP_TURBO)
    return depth_img


class  EvalNeRF:
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.data_dir = f'results/{self.hparams.dataset_name}/{self.hparams.exp_name}'
        self.data_dict = ['rgb_gt','rgb_pred','u_pred','depth']
        self.eval_dict = ['psnr', 'ssim', 'lpips']
        self.data = {key:[] for key in self.data_dict}
        self.evals = {key:[] for key in self.eval_dict}

    def extract_data(self):
        torch.cuda.empty_cache()
        data_files = sorted(glob.glob(os.path.join(self.data_dir, '*.npy')))
        for f in data_files:
            raw = np.load(f)
            for key in raw['data'].keys():
                self.data[key].append(raw['data'][key])
            for score in raw['eval'].keys():
                self.evals[score].append(raw['eval'][score])

        for key in self.data.keys():
            self.data[key] = np.array(self.data[key])
        for score in self.evals.keys():
            self.evals[score] = np.mean(self.evals[score])
        self.data['err'] = np.mean((self.data['rgb_gt'] - self.data['rgb_pred'])**2,-1)

    def compute_auc(self,intervals = 20):
        N_imgs = self.data['err'].shape[0]
        AUCs = []
        opts = []
        ROCs = []
        ROC_opts = []

        quants = [100. / intervals * t for t in range(1, intervals + 1)]
        for img_idx in range(N_imgs):
            ROC = []
            ROC_opt = []
            err = self.data['err'][img_idx].flatten()
            u_pred = self.data['u_pred'][img_idx].flatten()
            thresholds = [np.percentile(u_pred, q) for q in quants]
            opt_thresholds = [np.percentile(err, q) for q in quants]
            subs = [u_pred <= t for t in thresholds]
            opt_subs = [err <= t for t in opt_thresholds]
            ROC_points = [err[s].mean() for s in subs]
            ROC_opt_points = [err[s].mean() for s in opt_subs]

            [ROC.append(r) for r in ROC_points]
            [ROC_opt.append(r) for r in ROC_opt_points]
            AUC = np.trapz(ROC, dx=1. / intervals)
            opt = np.trapz(ROC_opt, dx=1. / intervals)

            ROCs.append(ROC)
            ROC_opts.append(ROC_opt)

            print(f'img_id: {img_idx:03d} \t AUC: {(AUC * 100.):.2f} \t opt: {(opt * 100.):.2f}')
            AUCs.append(AUC)
            opts.append(opt)

        avg_AUC = np.array(AUCs).mean()
        opt_AUC = np.array(opts).mean()

        self.evals['auc'] = avg_AUC * 100.
        self.evals['opt'] = opt_AUC * 100.
        self.data['ROC'] = np.array(ROCs).mean(-1)
        self.data['ROC_opt'] = np.array(ROC_opts).mean(-1)

        print('Total scene:')
        print('Avg. AUC: %f' % (avg_AUC * 100.))
        print('Opt. AUC: %f' % (opt_AUC * 100.))







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
    #system = NeRFSystem(hparams)

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
                               if hparams.num_gpus>1 else None,
                      num_sanity_val_steps=-1 if hparams.val_only else 0,
                      precision=16)

    if hparams.val_only:
        trainer.validate(model=system,dataloaders=system.val_dataloader())
        raise ValueError('Validation done!')
    else:
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
