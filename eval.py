from eval_opt import get_opts
import torch
import glob
import numpy as np
import os
import matplotlib.pyplot as plt
import imageio
from tqdm import tqdm
import time

# metrics
from torchmetrics import StructuralSimilarityIndexMeasure
from flip.flip_api import compute_ldrflip,color_space_transform
from train import err2img

import warnings; warnings.filterwarnings("ignore")

img2psnr = lambda x, y : 10 * np.log10(1.0 / (x - y)**2)

bhwc2bchw = lambda x: np.rollaxis(x, 3,1)

def img2flip(x,y):
    if x.ndim != y.ndim:
        raise ValueError('dim of x and y should match!')
    if x.ndim==3:
        test = color_space_transform(x, "linrgb2srgb")
        ref = color_space_transform(y, "linrgb2srgb")
        return compute_ldrflip(ref,test)  # 1xHxW
    if x.ndim==4:
        results = []
        for i in range(x.shape[0]):
            test = color_space_transform(x[i], "linrgb2srgb")
            ref = color_space_transform(y[i], "linrgb2srgb")
            results.append(compute_ldrflip(ref,test))
        return np.array(results).squeeze(1)


class  EvalNeRF:
    def __init__(self, hparams):
        super().__init__()

        # experiment settings
        self.hparams = hparams
        self.data_dir = self.hparams.val_dir
        self.scenes = [d for d in os.listdir(self.data_dir) if
                       os.path.isdir(d)] if not self.hparams.scenes else self.hparams.scenes
        if not isinstance(self.scenes, list):
            self.scenes = [self.scenes]
        self.data_dict = ['rgb_gt','rgb_pred','mcd','depth']
        self.data = {s:{} for s in self.scenes+['avg']}
        self.log_dir = self.data_dir if not self.hparams.log_dir else self.hparams.log_dir

        # metrics
        self.eval_dict = ['psnr', 'ssim', 'lpips']
        self.evals = {s: {} for s in self.scenes + ['avg']}
        self.psnr = img2psnr
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1,return_full_image=True)
        self.flip = img2flip

        # auc
        if not isinstance(self.hparams.est,list):
            self.hparams.est = [self.hparams.est]

    def extract_data(self,s):
        data = {key:[] for key in self.data_dict}
        evals = {key:[] for key in self.eval_dict}
        data_files = sorted(glob.glob(os.path.join(self.hparams.val_dir, s, '*.pth')))
        for f in data_files:
            raw = torch.load(f)
            for key in raw['data'].keys():
                data[key].append(raw['data'][key])
            for score in raw['eval'].keys():
                evals[score].append(raw['eval'][score])

        for key in data.keys():
            data[key] = np.array(data[key]) # BxHxWxC
        for score in evals.keys():
            evals[score] = np.mean(evals[score])
        data['err'] = np.mean((data['rgb_gt'] - data['rgb_pred'])**2,-1)
        self.data[s] = data
        self.evals[s] = evals
        self.evals[s]['N_frames'] = len(data_files)

    def compute_metric(self,s):
        rgb_pred = bhwc2bchw(self.data[s]['rgb_pred'])
        rgb_gt = bhwc2bchw(self.data[s]['rgb_gt'])
        if 'psnr' in self.hparams.est:
            self.data[s]['psnr'] = self.psnr(rgb_pred, rgb_gt).mean(1)  #
        if 'ssim' in self.hparams.est:
            _, batch_ssim = self.ssim(torch.from_numpy(rgb_pred),torch.from_numpy(rgb_gt))
            self.data[s]['ssim'] = batch_ssim.numpy().mean(1)
        if 'flip' in self.hparams.est:
            self.data[s]['flip'] = self.flip(rgb_pred, rgb_gt)

    def plot_metric(self,s):
        for key in self.hparams.est:
            if key=='flip':
                imgs = self.data[s][key]
                for idx in range(imgs.shape[0]):
                    imageio.imsave(os.path.join(self.hparams.val_dir, s, f'{idx:03d}_{key}.png'), err2img(imgs[idx]))
            if key in ['psnr','ssim']:
                imgs = self.data[s][key]
                for idx in range(imgs.shape[0]):
                    imageio.imsave(os.path.join(self.hparams.val_dir, s, f'{idx:03d}_{key}.png'), err2img(imgs[idx],flip=True))

    def compute_roc(self,opt,est,intervals = 20):
        ROC = []
        quants = [100. / intervals * t for t in range(1, intervals + 1)]
        thres = [np.percentile(est, q) for q in quants]
        subs = [est <= t for t in thres]
        ROC_points = [opt[s].mean() for s in subs]
        [ROC.append(r) for r in ROC_points]
        AUC = np.trapz(ROC, dx=1. / intervals)
        return AUC,ROC

    def compute_auc(self,s,intervals = 20):
        opts = self.data[s][self.hparams.opt]
        ests = {key: self.data[s][key] for key in self.hparams.est}
        N_imgs = opts.shape[0]
        AUC_ests = {key:[] for key in self.hparams.est}
        AUC_opts = []
        ROC_ests = {key:[] for key in self.hparams.est}
        ROC_opts = []

        for img_idx in range(N_imgs):
            opt = opts[img_idx].flatten()
            AUC_opt,ROC_opt = self.compute_roc(opt,opt,intervals)
            ROC_opts.append(ROC_opt)
            AUC_opts.append(AUC_opt)

            print(f'img_id: {img_idx:03d} \t AUC_opt: {(AUC_opt * 100.):.4f} ', end='\t')

            for key in self.hparams.est:
                est = ests[key][img_idx].flatten()
                if key in ['psnr', 'ssim']:
                    est = -est
                AUC_est, ROC_est = self.compute_roc(opt,est,intervals)
                ROC_ests[key].append(ROC_est)
                AUC_ests[key].append(AUC_est)

                print(f'AUC_{key}: {(AUC_est * 100.):.4f} ', end='\t')

            print('\n')

        avg_AUC = {key:np.array(AUC_ests[key]).mean()  for key in self.hparams.est}
        avg_AUC['opt'] = np.array(AUC_opts).mean()

        self.data[s]['ROC'] = {key: np.array(ROC_ests[key]).mean(0) for key in self.hparams.est}
        self.data[s]['ROC']['opt'] = np.array(ROC_opts).mean(0)
        self.evals[s]['auc'] = avg_AUC

        avg_opt = avg_AUC['opt']
        print(f'opt. AUC: {avg_opt * 100.:.4f}')
        for key in self.hparams.est:
            avg_auc = avg_AUC[key]
            print(f'Avg. {key} AUC: {(avg_auc * 100.):.4f}')

    def plot_roc(self,s):

        color = {'mcd': 'r',
                      'psnr': 'k',
                      'ssim': 'orange',
                      'flip': 'purple'}
        marker = {'mcd': "o",
                       'psnr': '^',
                       'ssim': 's',
                       'flip': '*'}

        quants = [100. / self.hparams.intervals * t for t in range(1, self.hparams.intervals + 1)]
        plt.figure()
        plt.plot(quants, self.data[s]['ROC']['opt'], marker="^",markersize=8, color='blue', label='opt')
        for key in self.hparams.est:
            plt.plot(quants, self.data[s]['ROC'][key], marker=marker[key],markersize=8, color=color[key], label=key)
        plt.xticks(quants)
        plt.xlabel('Sample Size(%)')
        plt.ylabel('Accumulative MSE')
        plt.legend()
        fig = plt.gcf()
        fig.set_size_inches(20, 8)
        plt.rcParams.update({'font.size': 20})
        if not s=='avg':
            fig.savefig(os.path.join(self.hparams.val_dir, s,'ROC_opt_vs_est.png'))
        else:
            fig.savefig(os.path.join(self.hparams.val_dir, 'ROC_opt_vs_est.png'))
        plt.close()

    def log_evals(self,s):
        if not s == 'avg':
            log_file = os.path.join(self.log_dir, s, self.hparams.log_file)
        else:
            log_file = os.path.join(self.log_dir, self.hparams.log_file)
        with open(log_file,'a') as f:
            f.write(f'scene={s}\n')
            for score in self.evals[s].keys():
                if not score=='auc':
                    f.write(f'{score}={self.evals[s][score]:.4f}\n')
                else:
                    for key in self.evals[s][score].keys():
                        f.write(f'{score}_{key}={self.evals[s][score][key]:.4f}\n')
            f.close()

    def run(self):

        start = time.time()

        avg_eval = {score:0 for score in self.eval_dict}
        avg_eval['auc'] = {key:0 for key in ['opt']+self.hparams.est}
        avg_roc = {key:[] for key in ['opt']+self.hparams.est}
        N_frames = 0

        print('Starting evalation......')

        for s in tqdm(self.scenes):
            self.extract_data(s)
            N_frames += self.evals[s]['N_frames']
            self.compute_metric(s)
            if self.hparams.plot_metric:
                self.plot_metric(s)

            self.compute_auc(s,self.hparams.intervals)
            if self.hparams.plot_roc:
                self.plot_roc(s)

            for score in self.eval_dict:
                avg_eval[score] += self.evals[s][score]*self.evals[s]['N_frames']
            for key in ['opt']+self.hparams.est:
                avg_eval['auc'][key] += self.evals[s]['auc'][key] * self.evals[s]['N_frames']
                avg_roc[key].append(self.data[s]['ROC'][key]*self.evals[s]['N_frames'])

        for key in self.eval_dict:
            avg_eval[key] /= N_frames
        for key in ['opt'] + self.hparams.est:
            avg_eval['auc'][key] /= N_frames

        self.evals['avg'] = avg_eval
        self.evals['avg']['N_frames'] = N_frames
        self.data['avg'] = {'ROC':avg_roc}

        for key in ['opt'] + self.hparams.est:
            self.data['avg']['ROC'][key] = np.array(avg_roc[key]).sum(0) / N_frames
        if self.hparams.plot_roc:
            self.plot_roc('avg')

        print('Saving results......')
        for s in tqdm(self.scenes+['avg']):
            self.log_evals(s)

        print('Done!')

        end = time.time()
        runtime = time.strftime("%H:%M:%S", time.gmtime(end - start))
        print('Total runtime: {}'.format(runtime))

if __name__ == '__main__':

    hparams = get_opts()
    EvalSystem = EvalNeRF(hparams)
    EvalSystem.run()

