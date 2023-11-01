import numpy as np
import torch
from einops import rearrange

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GetVirtualCam:
    def __init__(self, kwargs):
        super(GetVirtualCam, self).__init__()
        self.ref_c2w = kwargs['ref_c2w'] # 3x4
        self.K = kwargs['K'] # 3x3
        self.ref_depth_map = kwargs['ref_depth_map']
        self.device = kwargs['device']
        self.pixl_ids = kwargs['pix_ids']
        self.img_h = kwargs['img_h']
        self.img_w = kwargs['img_w']
        if kwargs.get('dense_map', False):
            self.ref_depth_map = rearrange(self.ref_depth_map, '(h w) -> h w', h=self.img_h),

        self.scene_center = self.get_scene_center()

    def get_scene_center(self):
        if self.ref_depth_map.ndim < 2:
            return self.get_scene_center_sparse()
        else:
            return self.get_scene_center_dense()

    def get_scene_center_dense(self):
        depth_map = self.ref_depth_map.clone().to(self.device)
        height, width = self.ref_depth_map.shape

        ref_c2w = torch.eye(4)
        ref_c2w[:3] = self.ref_c2w.clone().cpu()
        ref_c2w = ref_c2w.to(device=self.device, dtype=torch.float32)
        ref_w2c = torch.inverse(ref_c2w)

        K = torch.eye(4)
        K[:3, :3] = self.K.clone().cpu()
        K = K.to(ref_w2c)

        bwd_proj = torch.matmul(ref_c2w, torch.inverse(K)).to(torch.float32)
        bwd_rot = bwd_proj[:3, :3]
        bwd_trans = bwd_proj[:3, 3:4]

        y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32),
                               torch.arange(0, width, dtype=torch.float32)],
                              indexing='ij')
        y, x = y.contiguous(), x.contiguous()
        y, x = y.reshape(height * width), x.reshape(height * width)
        homog = torch.stack((x, y, torch.ones_like(x))).to(bwd_rot)

        # get world coords
        world_coords = torch.matmul(bwd_rot, homog)
        world_coords = world_coords * depth_map.reshape(1, -1)
        world_coords = world_coords + bwd_trans.reshape(3, 1)
        world_coords = torch.movedim(world_coords, 0, 1) # (h w) 3

        scene_center = world_coords.mean(0)

        return scene_center.cpu()

    def get_scene_center_sparse(self):
        depth_map = self.ref_depth_map.clone().to(self.device)
        height, width = self.img_h, self.img_w

        ref_c2w = torch.eye(4)
        ref_c2w[:3] = self.ref_c2w.clone().cpu()
        ref_c2w = ref_c2w.to(device=self.device, dtype=torch.float32)
        ref_w2c = torch.inverse(ref_c2w)

        K = torch.eye(4)
        K[:3, :3] = self.K.clone().cpu()
        K = K.to(ref_w2c)

        bwd_proj = torch.matmul(ref_c2w, torch.inverse(K)).to(torch.float32)
        bwd_rot = bwd_proj[:3, :3]
        bwd_trans = bwd_proj[:3, 3:4]

        y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32),
                               torch.arange(0, width, dtype=torch.float32)],
                              indexing='ij')
        y, x = y.contiguous(), x.contiguous()
        y, x = y.reshape(height * width), x.reshape(height * width)
        homog = torch.stack((x, y, torch.ones_like(x))).to(bwd_rot)
        homog = homog[...,self.pixl_ids]

        # get world coords
        world_coords = torch.matmul(bwd_rot, homog)
        world_coords = world_coords * depth_map.reshape(1, -1)
        world_coords = world_coords + bwd_trans.reshape(3, 1)
        world_coords = torch.movedim(world_coords, 0, 1) # (n_rays) 3

        scene_center = world_coords.mean(0)

        return scene_center.cpu()

    def get_near_c2w(self, c2w, theta=5, axis='x'):
        cam_center = c2w[:3, 3:4].clone().to(self.scene_center)
        cam_center = cam_center.squeeze()
        trans_c2s = self.get_translation_matrix(cam_center,self.scene_center)
        rot = self.get_rotation_matrix(theta, axis)

        c2w_homo = torch.eye(4)
        c2w_homo[:3] = c2w.clone().cpu()
        c2w_homo = c2w_homo.to(torch.float32)
        w2c = torch.inverse(c2w_homo)

        w2c = torch.mm(trans_c2s,w2c)
        w2c = torch.mm(rot,w2c)
        w2c = torch.mm(torch.inverse(trans_c2s),w2c)

        new_c2w = torch.inverse(w2c)
        return new_c2w[:3]

    def get_rotation_matrix(self, theta=5, axis='x'): # rot theta degree across x axis
        phi = (theta * (np.pi / 180.))
        rot = torch.eye(4)
        if axis=='x':
            rot[:3,:3] = torch.Tensor([
            [1, 0, 0],
            [0, np.cos(phi), -np.sin(phi)],
            [0, np.sin(phi), np.cos(phi)]
            ])
        elif axis == 'y':
            rot[:3,:3] = torch.Tensor([
                [np.cos(phi), 0, -np.sin(phi)],
                [0, 1, 0],
                [np.sin(phi), 0, np.cos(phi)]
            ])
        elif axis=='z':
            rot[:3,:3] = torch.Tensor([
                [np.cos(phi), -np.sin(phi), 0],
                [np.sin(phi), np.cos(phi), 0],
                [0, 0, 1],
            ])
        return rot

    def get_translation_matrix(self,origin,destination): # both should be (x,y,z)
        trans = torch.eye(4).to(destination)
        trans[:3,3] = destination-origin
        return trans