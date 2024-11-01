import numpy as np
import torch
from einops import rearrange

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from datasets.ray_utils import transform_pose, inv_transform_pose

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
        self.dense = kwargs['dense_map']
        self.opacity = kwargs['opacity']
        self.translate = kwargs['translate']
        self.scale = kwargs['scale']
        self.came_o = self.get_camera_center()

        if self.dense:
            self.ref_depth_map = self.ref_depth_map.reshape(self.img_h,self.img_w)

        self.scene_center = self.get_scene_center()

    def get_camera_center(self):
        origin_c2w = inv_transform_pose(self.ref_c2w, self.translate, self.scale)
        origin_R = origin_c2w[:3, :3]
        origin_T = origin_c2w[:3, 3]
        camera_o = -origin_R.T @ origin_T
        return camera_o

    def get_scene_center(self):
        if not self.dense:
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

        world_coords_real = world_coords[self.opacity>0]
        scene_center = world_coords_real.mean(0)

        return scene_center.cpu()

    def get_scene_center_sparse(self):
        depth_map = self.ref_depth_map.clone().to(self.device)
        if len(depth_map) > len(self.pixl_ids):
            depth_map = depth_map[self.pixl_ids]
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

        world_coords_real = world_coords[self.opacity > 0]
        scene_center = world_coords_real.mean(0)

        return scene_center.cpu()

    def random_points_on_sphere(self, N, r, O):
        """
        Generate N random points on a sphere of radius r centered at O.

        Args:
        - N: number of points
        - r: radius of the sphere
        - O: center of the sphere as a tensor of shape (3,), i.e., O = torch.tensor([x, y, z])

        Returns:
        - points: a tensor of shape (N, 3) representing the N random points on the sphere
        """
        points = torch.rand(N,3).to(O)
        points = 2*points-torch.ones_like(points)
        points = points / torch.norm(points, dim=1, keepdim=True)
        points = points * r + O

        return points

    def get_N_near_c2w(self, N, radiaus_ratio=0.1):
        radiaus = self.ref_depth_map.flatten().median() * radiaus_ratio
        new_translations = self.random_points_on_sphere(N, radiaus, self.ref_c2w[:3,3])
        vir_c2ws = []
        look_at = self.scene_center
        for new_t in new_translations:
            # create new camera pose by look at
            forward = look_at - new_t
            forward /= torch.linalg.norm(forward)
            forward = forward.to(new_t)
            # world_up = torch.tensor(self.center_camera.metadata['world_up']).to(new_t.device)

            world_up = torch.tensor([0., 1., 0.]).to(new_t.device)  # need to be careful for the openGL system!!!
            right = torch.cross(world_up, forward)
            right /= torch.linalg.norm(right)
            # left = -right
            up = torch.cross(forward, right)
            new_R = torch.vstack([right, up, forward]).T
            # new_R = torch.vstack([left, up, forward]).T
            # new_T = -new_R @ new_o
            new_T = new_t

            new_c2w = torch.eye(4).to(new_t.device)
            new_c2w[:3, :3] = new_R
            new_c2w[:3, 3] = new_T

            new_c2w = transform_pose(new_c2w, self.translate,self.scale)
            vir_c2ws.append(new_c2w)

        return vir_c2ws

    def spheric_pose(self,theta, phi, radius, mean_h):
        trans_t = lambda t : np.array([
            [1,0,0,0],
            [0,1,0,2*mean_h],
            [0,0,1,-t]
        ])

        rot_phi = lambda phi : np.array([
            [1,0,0],
            [0,np.cos(phi),-np.sin(phi)],
            [0,np.sin(phi), np.cos(phi)]
        ])

        rot_theta = lambda th : np.array([
            [np.cos(th),0,-np.sin(th)],
            [0,1,0],
            [np.sin(th),0, np.cos(th)]
        ])

        c2w = rot_theta(theta) @ rot_phi(phi) @ trans_t(radius)
        c2w = np.array([[-1,0,0],[0,0,1],[0,1,0]]) @ c2w
        return c2w

    def get_N_near_spheric_poses(self, radius, mean_h, n_poses=6):
        """
        Create circular poses around z axis.
        Inputs:
            radius: the (negative) height and the radius of the circle.
            mean_h: mean camera height
        Outputs:
            spheric_poses: (n_poses, 3, 4) the poses in the circular path
        """
        spheric_poses = []
        ths = np.random.randint(0,100,n_poses)*(2*np.pi/100)
        ths = ths.tolist()
        for th in ths:
            spheric_poses += [self.spheric_pose(th, -np.pi/12, radius, mean_h)]
        return np.stack(spheric_poses, 0)

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