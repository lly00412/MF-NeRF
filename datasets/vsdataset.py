from torch.utils.data import Dataset
import numpy as np
import pandas as pd


class VSDataset(Dataset):
    """
    Define length and sampling method
    """
    def __init__(self, data_file, split='train', scene='Hotdog'):
        self.data_file = data_file
        self.split = split
        self.scene = scene
        self.random_seed = 758669
        self.read_raw_data()

    def read_raw_data(self):
        raw_df = pd.read_csv(self.data_file)
        df = raw_df[raw_df['scene'] == self.scene]
        n_views = len(df)
        features = []
        labels = []
        for i in range(n_views):
            for j in range(n_views):
                if i

    def __len__(self):
        if self.split.startswith('train'):
            return 1000
        return len(self.poses)

    def __getitem__(self, idx):
        if self.split.startswith('train'):
            # training pose is retrieved in train.py
            if self.ray_sampling_strategy == 'all_images': # randomly select images
                img_idxs = np.random.choice(len(self.poses), self.batch_size)
            elif self.ray_sampling_strategy == 'same_image': # randomly select ONE image
                img_idxs = np.random.choice(len(self.poses), 1)[0]
            elif self.ray_sampling_strategy == 'weighted_images': # new coming img has higher probability
                img_idxs = np.random.choice(len(self.poses), self.batch_size,p=self.p)
            # randomly select pixels
            pix_idxs = np.random.choice(self.img_wh[0]*self.img_wh[1], self.batch_size, replace=False)
            rays = self.rays[img_idxs, pix_idxs]
            sample = {'img_idxs': img_idxs, 'pix_idxs': pix_idxs,
                      'rgb': rays[:, :3]}
            if self.rays.shape[-1] == 4: # HDR-NeRF data
                sample['exposure'] = rays[:, 3:]
        else:
            sample = {'pose': self.poses[idx], 'img_idxs': idx}
            if hasattr(self, "raw_poses"):
                sample['raw_pose'] = self.raw_poses[idx]
            if len(self.rays)>0: # if ground truth available
                rays = self.rays[idx]
                sample['rgb'] = rays[:, :3]
                if rays.shape[1] == 4: # HDR-NeRF data
                    sample['exposure'] = rays[0, 3] # same exposure for all rays

        return sample