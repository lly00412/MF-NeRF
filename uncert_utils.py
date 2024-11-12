import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from typing import Tuple

def enable_dropout(model,p=0.2):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()
            m.p=p

def close_dropout(model):
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.eval()

class Projection(nn.Module):
    """Layer which projects 3D points into a camera view
    """
    def __init__(self, height, width, eps=1e-7):
        super(Projection, self).__init__()

        self.height = height
        self.width = width
        self.eps = eps

    def forward(self, points3d, K, normalized=True):
        """
        Args:
            points3d (torch.tensor, [N,4,(HxW)]: 3D points in homogeneous coordinates
            K (torch.tensor, [torch.tensor, (N,4,4)]: camera intrinsics
            normalized (bool):
                - True: normalized to [-1, 1]
                - False: [0, W-1] and [0, H-1]
        Returns:
            xy (torch.tensor, [N,H,W,2]): pixel coordinates
        """
        # projection
        points2d = torch.matmul(K[:, :3, :], points3d)

        # convert from homogeneous coordinates
        xy = points2d[:, :2, :] / (points2d[:, 2:3, :] + self.eps)
        xy = xy.view(points3d.shape[0], 2, self.height, self.width)
        xy = xy.permute(0, 2, 3, 1)

        # normalization
        if normalized:
            xy[..., 0] /= self.width - 1
            xy[..., 1] /= self.height - 1
            xy = (xy - 0.5) * 2
        return xy

class Transformation3D(nn.Module):
    """Layer which transform 3D points
    """
    def __init__(self):
        super(Transformation3D, self).__init__()

    def forward(self,
                points: torch.Tensor,
                T: torch.Tensor
                ) -> torch.Tensor:
        """
        Args:
            points (torch.Tensor, [N,4,(HxW)]): 3D points in homogeneous coordinates
            T (torch.Tensor, [N,4,4]): transformation matrice
        Returns:
            transformed_points (torch.Tensor, [N,4,(HxW)]): 3D points in homogeneous coordinates
        """
        transformed_points = torch.matmul(T, points)
        return transformed_points

class Backprojection(nn.Module):
    """Layer to backproject a depth image given the camera intrinsics

    Attributes
        xy (torch.tensor, [N,3,HxW]: homogeneous pixel coordinates on regular grid
    """

    def __init__(self, height, width):
        """
        Args:
            height (int): image height
            width (int): image width
        """
        super(Backprojection, self).__init__()

        self.height = height
        self.width = width

        # generate regular grid
        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        id_coords = torch.tensor(id_coords)

        # generate homogeneous pixel coordinates
        self.ones = nn.Parameter(torch.ones(1, 1, self.height * self.width),
                                 requires_grad=False)
        self.xy = torch.unsqueeze(
            torch.stack([id_coords[0].view(-1), id_coords[1].view(-1)], 0)
            , 0)
        self.xy = torch.cat([self.xy, self.ones], 1)
        self.xy = nn.Parameter(self.xy, requires_grad=False)

    def forward(self, depth, inv_K, img_like_out=False):
        """
        Args:
            depth (torch.tensor, [N,1,H,W]: depth map
            inv_K (torch.tensor, [N,4,4]): inverse camera intrinsics
            img_like_out (bool): if True, the output shape is [N,4,H,W]; else [N,4,(HxW)]
        Returns:
            points (torch.tensor, [N,4,(HxW)]): 3D points in homogeneous coordinates
        """
        depth = depth.contiguous()

        xy = self.xy.repeat(depth.shape[0], 1, 1)
        ones = self.ones.repeat(depth.shape[0], 1, 1)

        points = torch.matmul(inv_K[:, :3, :3], xy)
        points = depth.view(depth.shape[0], 1, -1) * points
        points = torch.cat([points, ones], 1)

        if img_like_out:
            points = points.reshape(depth.shape[0], 4, self.height, self.width)
        return points

class BackwardWarping(nn.Module):

    def __init__(self,
                 out_hw: Tuple[int,int],
                 device: torch.device,
                 K:torch.Tensor) -> None:
        super(BackwardWarping,self).__init__()
        height, width = out_hw
        self.backproj = Backprojection(height,width).to(device)
        self.projection = Projection(height,width).to(device)
        self.transform3d = Transformation3D().to(device)

        H,W = height,width
        self.rgb = torch.zeros(H,W,3).view(-1,3).to(device)
        self.depth = torch.zeros(H, W, 1).view(-1, 1).to(device)
        self.K = K.to(device)
        self.inv_K = torch.inverse(K).to(device)
        self.K = self.K.unsqueeze(0)
        self.inv_K = self.inv_K.unsqueeze(0) # 1,4,4
    def forward(self,
                img_src: torch.Tensor,
                depth_src: torch.Tensor,
                depth_tgt: torch.Tensor,
                tgt2src_transform: torch.Tensor,
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, _, h, w = depth_tgt.shape

        # reproject
        pts3d_tgt = self.backproj(depth_tgt,self.inv_K)
        pts3d_src = self.transform3d(pts3d_tgt,tgt2src_transform)
        src_grid = self.projection(pts3d_src,self.K,normalized=True)
        transformed_distance = pts3d_src[:, 2:3].view(b,1,h,w)

        img_tgt = F.grid_sample(img_src, src_grid, mode = 'bilinear', padding_mode = 'zeros')
        depth_src2tgt = F.grid_sample(depth_src, src_grid, mode='bilinear', padding_mode='zeros')

        # rm invalid depth
        valid_depth_mask = (transformed_distance < 1e6) & (depth_src2tgt > 0)

        # rm invalid coords
        vaild_coord_mask = (src_grid[...,0]> -1) & (src_grid[...,0] < 1) & (src_grid[...,1]> -1) & (src_grid[...,1] < 1)
        vaild_coord_mask = vaild_coord_mask.unsqueeze(1)

        valid_mask = valid_depth_mask & vaild_coord_mask
        invaild_mask = ~valid_mask

        return img_tgt.float(), depth_src2tgt.float(), invaild_mask.float()


def warp_tgt_to_ref(tgt_depth, ref_c2w, tgt_c2w, K, pixl_ids=None, img_shape=None,device='cpu'):
    depth_map = tgt_depth.clone()
    height, width = img_shape

    K_homo = torch.eye(4)
    K_homo[:3,:3] = K.clone().cpu()

    rc2w = torch.eye(4)
    rc2w[:3] = ref_c2w.clone().cpu()
    rw2c = torch.inverse(rc2w)

    tc2w = torch.eye(4)
    tc2w[:3] = tgt_c2w.clone().cpu()
    tw2c = torch.inverse(tc2w)

    torch.cuda.empty_cache()
    # warp tgt depth map to ref view
    # grab intrinsics and extrinsics from reference view
    P_ref = rw2c.to(torch.float32) # 4x4
    K_ref = K_homo.clone().to(torch.float32) # 4x4

    P_ref = P_ref.to(device)
    K_ref = K_ref.to(device)

    R_ref = P_ref[:3, :3] # 3x3
    t_ref = P_ref[:3, 3:4] # 3x1

    C_ref = torch.matmul(-R_ref.transpose(0, 1), t_ref)
    z_ref = R_ref[2:3, :3].reshape(1, 1, 1, 3).repeat(height, width, 1, 1)
    C_ref = C_ref.reshape(1, 1, 3).repeat(height, width, 1)

    depth_map = depth_map.to(device)  # h,w

    # get intrinsics and extrinsics from target view
    P_tgt = tw2c.to(torch.float32)  #  4x4
    K_tgt = K_homo.clone().to(torch.float32)  #  4x4

    P_tgt = P_tgt.to(device)
    K_tgt = K_tgt.to(device)

    bwd_proj = torch.matmul(torch.inverse(P_tgt), torch.inverse(K_tgt)).to(torch.float32)
    fwd_proj = torch.matmul(K_ref, P_ref).to(torch.float32)
    bwd_rot = bwd_proj[:3, :3]
    bwd_trans = bwd_proj[:3, 3:4]
    proj = torch.matmul(fwd_proj, bwd_proj)
    rot = proj[:3, :3]
    trans = proj[:3, 3:4]

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
    world_coords = torch.movedim(world_coords, 0, 1)
    world_coords = world_coords.reshape(height, width, 3)

    # get pixel projection
    rot_coords = torch.matmul(rot, homog)
    proj_3d = rot_coords * depth_map.reshape(1, -1)
    proj_3d = proj_3d + trans.reshape(3, 1)
    proj_2d = proj_3d[:2, :] / proj_3d[2:3, :]
    proj_2d = (torch.movedim(proj_2d, 0, 1)).to(torch.long)
    proj_2d = torch.flip(proj_2d, dims=(1,))

    # compute projected depth
    proj_depth = torch.sub(world_coords, C_ref).unsqueeze(-1)
    proj_depth = torch.matmul(z_ref, proj_depth).reshape(height, width)
    proj_depth = proj_depth.reshape(-1, 1)

    # mask out invalid indices
    mask = torch.where(proj_2d[:, 0] < height, 1, 0) * \
           torch.where(proj_2d[:, 0] >= 0, 1, 0) * \
           torch.where(proj_2d[:, 1] < width, 1, 0) * \
           torch.where(proj_2d[:, 1] >= 0, 1, 0)
    inds = torch.where(mask)[0]
    proj_2d = torch.index_select(proj_2d, dim=0, index=inds)
    proj_2d = (proj_2d[:, 0] * width) + proj_2d[:, 1]
    proj_depth = torch.index_select(proj_depth, dim=0, index=inds).squeeze()

    proj_depth, indices = torch.sort(proj_depth, 0) # ascending oreder
    proj_2d = proj_2d[indices]
    proj_depth = proj_depth.flip(0)
    proj_2d = proj_2d.flip(0)

    warped_depth = torch.zeros(height*width).to(proj_depth)
    warped_depth[proj_2d] = proj_depth
    warped_depth = warped_depth.reshape(height, width)

    del proj_depth

    return warped_depth, proj_2d

def warp_tgt_to_ref_sparse(tgt_depth, ref_c2w, tgt_c2w, K, pixl_ids, img_shape, device='cpu'):
    torch.cuda.empty_cache()

    depth_map = tgt_depth.clone()  # (N_rays)
    if len(depth_map)> len(pixl_ids):
        depth_map = tgt_depth[pixl_ids]
    else:
        depth_map = tgt_depth

    height, width = img_shape
    n_rays = depth_map.shape[0]

    K_homo = torch.eye(4)
    K_homo[:3,:3] = K.clone().cpu()

    rc2w = torch.eye(4)
    rc2w[:3] = ref_c2w.clone().cpu()
    rw2c = torch.inverse(rc2w)

    tc2w = torch.eye(4)
    tc2w[:3] = tgt_c2w.clone().cpu()
    tw2c = torch.inverse(tc2w)

    # warp tgt depth map to ref view
    # grab intrinsics and extrinsics from reference view
    P_ref = rw2c.to(torch.float32) # 4x4
    K_ref = K_homo.clone().to(torch.float32) # 4x4

    P_ref = P_ref.to(device)
    K_ref = K_ref.to(device)

    R_ref = P_ref[:3, :3] # 3x3
    t_ref = P_ref[:3, 3:4] # 3x1

    C_ref = torch.matmul(-R_ref.transpose(0, 1), t_ref)
    # z_ref = R_ref[2:3, :3].reshape(1, 1, 1, 3).repeat(height, width, 1, 1)
    # C_ref = C_ref.reshape(1, 1, 3).repeat(height, width, 1)

    z_ref = R_ref[2:3, :3].reshape(1, 1, 3).repeat(n_rays, 1, 1)
    C_ref = C_ref.reshape(1, 3).repeat(n_rays, 1)


    depth_map = depth_map.to(device)  # h,w

    # get intrinsics and extrinsics from target view
    P_tgt = tw2c.to(torch.float32)  #  4x4
    K_tgt = K_homo.clone().to(torch.float32)  #  4x4

    P_tgt = P_tgt.to(device)
    K_tgt = K_tgt.to(device)

    #bwd_proj = torch.matmul(torch.inverse(P_tgt), torch.inverse(K_tgt)).to(torch.float32)
    # fwd_proj = torch.matmul(K_ref, P_ref).to(torch.float32)
    bwd_proj = torch.inverse(P_tgt) @ torch.inverse(K_tgt)
    fwd_proj = K_ref @ P_ref
    # breakpoint()
    bwd_rot = bwd_proj[:3, :3]
    bwd_trans = bwd_proj[:3, 3:4]
    # proj = torch.matmul(fwd_proj, bwd_proj)
    proj = fwd_proj @ bwd_proj
    proj = proj.to(torch.float32)
    rot = proj[:3, :3]
    trans = proj[:3, 3:4]

    y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32),
                           torch.arange(0, width, dtype=torch.float32)],
                          indexing='ij')
    y, x = y.contiguous(), x.contiguous()
    y, x = y.reshape(height * width), x.reshape(height * width)
    homog = torch.stack((x, y, torch.ones_like(x))).to(bwd_rot)
    homog = homog[..., pixl_ids] # (N_rays, 3)

    # get world coords
    # world_coords = torch.matmul(bwd_rot, homog)  # (N_rays, 3)
    world_coords = bwd_rot @ homog
    world_coords = world_coords * depth_map.reshape(1, -1)
    world_coords = world_coords + bwd_trans.reshape(3, 1)
    world_coords = torch.movedim(world_coords, 0, 1)  # (N_rays, 3)
    # world_coords = world_coords.reshape(height, width, 3)

    # get pixel projection
    # rot_coords = torch.matmul(rot, homog)
    rot_coords = rot @ homog
    proj_3d = rot_coords * depth_map.reshape(1, -1)
    proj_3d = proj_3d + trans.reshape(3, 1)
    proj_2d = proj_3d[:2, :] / proj_3d[2:3, :]
    proj_2d = (torch.movedim(proj_2d, 0, 1)).to(torch.long)
    proj_2d = torch.flip(proj_2d, dims=(1,))

    # compute projected depth
    proj_depth = torch.sub(world_coords, C_ref).unsqueeze(-1)
    # proj_depth = torch.matmul(z_ref, proj_depth)
    proj_depth = z_ref @ proj_depth
    proj_depth = proj_depth.reshape(-1, 1)

    # mask out invalid indices
    mask = torch.where(proj_2d[:, 0] < height, 1, 0) * \
           torch.where(proj_2d[:, 0] >= 0, 1, 0) * \
           torch.where(proj_2d[:, 1] < width, 1, 0) * \
           torch.where(proj_2d[:, 1] >= 0, 1, 0)


    pixl_ids = proj_2d[:, 0] * width + proj_2d[:, 1]
    pixl_ids[mask == 0] = 0
    pixl_ids = pixl_ids.squeeze(-1)

    proj_depth, indices = torch.sort(proj_depth, 0)  # ascending oreder
    sorted_pixl_ids = pixl_ids[indices]
    proj_depth = proj_depth.flip(0)
    sorted_pixl_ids = sorted_pixl_ids.flip(0)

    warped_depth = torch.zeros(height * width).to(proj_depth)
    warped_depth[sorted_pixl_ids] = proj_depth
    warped_depth = warped_depth[pixl_ids]

    del proj_depth,sorted_pixl_ids,indices

    warped_depth[pixl_ids==0] = 0  # (N_rays)
    warped_depth = warped_depth.squeeze(-1)

    # mask = mask.reshape(-1, 1).to(proj_depth)
    # warped_depth = proj_depth * mask  # (N_rays)
    # warped_depth = warped_depth.squeeze(-1)

    # proj_2d[:, 0] = torch.where(proj_2d[:, 0] >= height, height-1, proj_2d[:, 0])
    # proj_2d[:, 0] = torch.where(proj_2d[:, 0] < 0, 0, proj_2d[:, 0])
    # proj_2d[:, 1] = torch.where(proj_2d[:, 1] >= width, width-1, proj_2d[:, 1])
    # proj_2d[:, 1] = torch.where(proj_2d[:, 1] < 0, 0, proj_2d[:, 1])

    return warped_depth, pixl_ids.long()

def warp_tgt_to_ref_sparse_v2(tgt_depth, ref_c2w, tgt_c2w, K, pixl_ids, img_shape, device='cpu'):
    torch.cuda.empty_cache()

    depth_map = tgt_depth.clone()  # (N_rays)
    if len(depth_map)> len(pixl_ids):
        depth_map = tgt_depth[pixl_ids]
    else:
        depth_map = tgt_depth

    height, width = img_shape
    n_rays = depth_map.shape[0]

    K_homo = torch.eye(4)
    K_homo[:3,:3] = K.clone().cpu()

    rc2w = torch.eye(4)
    rc2w[:3] = ref_c2w.clone().cpu()
    rw2c = torch.inverse(rc2w)

    tc2w = torch.eye(4)
    tc2w[:3] = tgt_c2w.clone().cpu()
    tw2c = torch.inverse(tc2w)

    # warp tgt depth map to ref view
    # grab intrinsics and extrinsics from reference view
    P_ref = rw2c.to(torch.float32) # 4x4
    K_ref = K_homo.clone().to(torch.float32) # 4x4

    P_ref = P_ref.to(device)
    K_ref = K_ref.to(device)

    R_ref = P_ref[:3, :3] # 3x3
    t_ref = P_ref[:3, 3:4] # 3x1

    C_ref = torch.matmul(-R_ref.transpose(0, 1), t_ref)
    # z_ref = R_ref[2:3, :3].reshape(1, 1, 1, 3).repeat(height, width, 1, 1)
    # C_ref = C_ref.reshape(1, 1, 3).repeat(height, width, 1)

    z_ref = R_ref[2:3, :3].reshape(1, 1, 3).repeat(n_rays, 1, 1)
    C_ref = C_ref.reshape(1, 3).repeat(n_rays, 1)

    depth_map = depth_map.to(device)  # h,w

    # get intrinsics and extrinsics from target view
    P_tgt = tw2c.to(torch.float32)  #  4x4
    K_tgt = K_homo.clone().to(torch.float32)  #  4x4

    P_tgt = P_tgt.to(device)
    K_tgt = K_tgt.to(device)

    #bwd_proj = torch.matmul(torch.inverse(P_tgt), torch.inverse(K_tgt)).to(torch.float32)
    # fwd_proj = torch.matmul(K_ref, P_ref).to(torch.float32)
    bwd_proj = torch.inverse(P_tgt) @ torch.inverse(K_tgt)
    fwd_proj = K_ref @ P_ref
    # breakpoint()
    bwd_rot = bwd_proj[:3, :3]
    bwd_trans = bwd_proj[:3, 3:4]
    # proj = torch.matmul(fwd_proj, bwd_proj)
    proj = fwd_proj @ bwd_proj
    proj = proj.to(torch.float32)
    rot = proj[:3, :3]
    trans = proj[:3, 3:4]

    y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32),
                           torch.arange(0, width, dtype=torch.float32)],
                          indexing='ij')
    y, x = y.contiguous(), x.contiguous()
    y, x = y.reshape(height * width), x.reshape(height * width)
    homog = torch.stack((x, y, torch.ones_like(x))).to(bwd_rot)
    homog = homog[..., pixl_ids] # (N_rays, 3)

    bwd_proj_tgt = torch.inverse(K_tgt[:3,:3]) @ homog
    proj_3d = bwd_proj_tgt * depth_map
    ones = torch.ones_like(depth_map).unsqueeze(0)
    proj_3d_homo = torch.cat([proj_3d, ones], 0)
    tgt_to_ref = torch.inverse(P_ref) @ P_tgt @ proj_3d_homo
    fwd_proj_ref = K_ref @ tgt_to_ref
    proj_2d = fwd_proj_ref[:3,...]
    proj_2d[:2,...] = proj_2d[:2, ...] / proj_2d[2:3, ...]

    proj_2d = proj_2d.long()
    proj_2d[0, :] = torch.where(proj_2d[0, :] >= width, width-1, proj_2d[0, :])
    proj_2d[0, :] = torch.where(proj_2d[0, :] < 0, 0, proj_2d[0, :])
    proj_2d[1, :] = torch.where(proj_2d[1, :] >= height, height-1, proj_2d[1, :])
    proj_2d[1, :] = torch.where(proj_2d[1, :] < 0, 0, proj_2d[1, :])
    pixl_ids = proj_2d[0, :] * height + proj_2d[1, :]

    return pixl_ids

def compute_roc(opt,est,intervals = 10): # input numpy array
    ROC = []
    quants = [100. / intervals * t for t in range(1, intervals + 1)]
    thres = [np.percentile(est, q) for q in quants]
    subs = [est <= t for t in thres]
    ROC_points = [opt[s].mean() for s in subs]
    [ROC.append(r) for r in ROC_points]
    AUC = np.trapz(ROC, dx=1. / intervals)
    return ROC,AUC

def compute_ause(opt,est,intervals = 10): # input torch.tensor
    quants = [100. / intervals * t for t in range(1, intervals + 1)]
    thres = [np.percentile(est, q) for q in quants]

    opt_roc = []
    opt_subs = [opt <= t for t in thres]
    opt_points = [opt[s].mean() if s.any() else 0.0 for s in opt_subs]
    opt_roc.extend(opt_points)
    opt_roc = np.array(opt_roc)

    est_roc = []
    est_subs = [est <= t for t in thres]
    est_points = [opt[s].mean() if s.any() else 0.0 for s in est_subs]
    est_roc.extend(est_points)
    est_roc = np.array(est_roc)

    max_val = opt_roc.max()
    opt_roc_norm = opt_roc / max_val
    est_roc_norm = est_roc / max_val
    ause = np.trapz(np.abs(est_roc_norm-opt_roc_norm), dx=1.0 / intervals).item()

    return est_roc,ause


method_dict={
   'warp' : 'VS-NeRF',
    'mcd_d': 'MCD-Depth',
    'mcd_r':'MCD-RGB',
    'rgb_err': 'RGB_Error',
    'entropy': 'Entropy'
}

def plot_roc(ROC_dict,fig_name, opt_label='rgb_err',intervals = 10):
    quants = [100. / intervals * t for t in range(1, intervals + 1)]
    plt.figure()
    plt.rcParams.update({'font.size': 25})
    # plot opt
    ROC_opt = ROC_dict.pop(opt_label)
    ymax = max(ROC_opt) * 1.2
    ymin = 0. - max(ROC_opt) * 0.1
    plt.plot(quants, ROC_opt, marker="^", markersize=10, linewidth= 2,color='blue', label=method_dict[opt_label])
    for est in ROC_dict.keys():
        if est in method_dict.keys():
            est_name = method_dict[est]
        else:
            est_name = f'theta = {est}'
        plt.plot(quants, ROC_dict[est], marker="o", markersize=10,linewidth= 2, label=est_name)
    xlabels = [100. / 10 * t for t in range(1, 10 + 1)]
    plt.xticks(xlabels)
    plt.xlabel('Sample Size(%)')
    plt.ylabel('AUSE')
    plt.ylim([ymin,ymax])
    plt.legend()
    fig = plt.gcf()
    fig.set_size_inches(20, 12)
    fig.savefig(fig_name)
    plt.close()




