import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GetVirtualCam:
    def __init__(self, kwargs):
        super(GetVirtualCam, self).__init__()
        self.ref_c2w = kwargs['ref_c2w'] # 3x4
        self.K = kwargs['K'] # 3x3
        self.ref_depth_map = kwargs['ref_depth_map']
        self.device = kwargs['device']

        self.scene_center = self.get_scene_center()

    def __call__(self, c2w, all_poses=None, j=None, iter_=None):
        assert (c2w.shape == (3, 4))

        if self.near_c2w_type == 'rot_from_origin':
            return self.rot_from_origin(c2w, iter_)
        elif self.near_c2w_type == 'near':
            return self.near(c2w, all_poses)
        elif self.near_c2w_type == 'random_pos':
            return self.random_pos(c2w)
        elif self.near_c2w_type == 'random_dir':
            return self.random_dir(c2w, j)

    def get_scene_center(self):
        depth_map = self.ref_depth_map.clone().to(self.device)
        height, width = self.ref_depth_map.shape

        ref_c2w = torch.eye(4)
        ref_c2w[:3] = self.ref_c2w.cpu().clone()
        ref_c2w = ref_c2w.to(device=self.device, dtype=torch.float32)
        ref_w2c = torch.inverse(ref_c2w)

        K = torch.eye(4)
        K[:3, :3] = self.K.cpu().clone()
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

        return scene_center

    def get_near_c2w(self, c2w, theta=5, axis='x'):
        cam_center = c2w[:3, 3:4].to(self.scene_center)
        trans_c2s = self.get_translation_matrix(cam_center,self.scene_center)
        rot = self.get_rotation_matrix(theta, axis)

        c2w_homo = torch.eye(4)
        c2w_homo[:3] = c2w.cpu().clone()
        c2w_homo = c2w_homo.to(torch.float32)
        w2c = torch.inverse(c2w_homo)

        w2c = torch.mm(trans_c2s,w2c)
        w2c = torch.mm(rot,w2c)
        w2c = torch.mm(torch.inverse(trans_c2s),w2c)

        new_c2w = torch.inverse(w2c)
        return new_c2w[:3]

    def get_rotation_matrix(self, theta=5, axis='x'): # rot theta degree across x axis
        phi = (theta * (torch.pi / 180.))
        rot = torch.eye(4)
        if axis=='x':
            rot[:3,:3] = torch.Tensor([
            [1, 0, 0],
            [0, torch.cos(phi), -torch.sin(phi)],
            [0, torch.sin(phi), torch.cos(phi)]
            ])
        elif axis == 'y':
            rot[:3,:3] = torch.Tensor([
                [torch.cos(phi), 0, -torch.sin(phi)],
                [0, 1, 0],
                [torch.sin(phi), 0, torch.cos(phi)]
            ])
        elif axis=='z':
            rot[:3,:3] = torch.Tensor([
                [torch.cos(phi), -torch.sin(phi), 0],
                [torch.sin(phi), torch.cos(phi), 0],
                [0, 0, 1],
            ])
        return rot

    def get_translation_matrix(self,origin,destination): # both should be (x,y,z)
        trans = torch.eye(4)
        trans[:3,3] = destination-origin
        return trans