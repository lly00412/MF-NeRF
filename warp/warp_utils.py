import numpy as np
from .build.render_tgt_volume import render_to_ref

def warp_tgt_to_ref(tgt_depth, ref_cams, tgt_cams):
    rcams = np.stack([np.eye(4),np.eye(4)])
    rcams[0][:3] = ref_cams['pose']
    rcams[1][:3,:3] = ref_cams['K']

    tcams = np.stack([np.eye(4),np.eye(4)])
    tcams[0][:3] = tgt_cams['pose']
    tcams[1][:3, :3] = tgt_cams['K']

    warp_depth = render_to_ref(tgt_depth.shape,tgt_depth.flatten().tolist(),rcams.tolist(),tcams.tolist())

    return warp_depth


# def warp_tgt_to_ref(tgt_depth, ref_cams, tgt_cams, device='cpu'):
#     # warp tgt depth map to ref view
#     height, width = tgt_depth.shape
#     # grab intrinsics and extrinsics from reference view
#     P_ref = torch.zeros(4,4)
#     P_ref[:3,:] = ref_cams['pose'] # 3x4 -> 4x4
#     P_ref[-1,-1] = 1
#
#     K_ref = torch.zeros(4,4)
#     K_ref[:3,:3] = ref_cams['K'] # 3x3 -> 4x4
#     K_ref[-1, -1] = 1
#
#     P_ref = P_ref.to(device)
#     K_ref = K_ref.to(device)
#
#     R_ref = P_ref[:3, :3] # 3x3
#     t_ref = P_ref[:3, 3:4] # 3x1
#
#     C_ref = torch.matmul(-R_ref.transpose(0, 1), t_ref)
#     z_ref = R_ref[2:3, :3].reshape(1, 1, 1, 3).repeat(height, width, 1, 1)
#     C_ref = C_ref.reshape(1, 1, 3).repeat(height, width, 1)
#
#     depth_map = tgt_depth.to(device)  # h,w
#
#     # get intrinsics and extrinsics from target view
#     P_tgt = torch.zeros(4, 4)
#     P_tgt[:3, :] = tgt_cams['pose']  # 3x4 -> 4x4
#     P_tgt[-1, -1] = 1
#
#     K_tgt = torch.zeros(4, 4)
#     K_tgt[:3, :3] = tgt_cams['K']  # 3x3 -> 4x4
#     K_tgt[-1, -1] = 1
#
#     P_tgt = P_tgt.to(device)
#     K_tgt = K_tgt.to(device)
#
#     bwd_proj = torch.matmul(torch.inverse(P_tgt), torch.inverse(K_tgt)).to(torch.float32)
#     fwd_proj = torch.matmul(K_ref, P_ref).to(torch.float32)
#     bwd_rot = bwd_proj[:3, :3]
#     bwd_trans = bwd_proj[:3, 3:4]
#     proj = torch.matmul(fwd_proj, bwd_proj)
#     rot = proj[:3, :3]
#     trans = proj[:3, 3:4]
#
#     y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32),
#                            torch.arange(0, width, dtype=torch.float32)],
#                           indexing='ij')
#     y, x = y.contiguous(), x.contiguous()
#     y, x = y.reshape(height * width), x.reshape(height * width)
#     homog = torch.stack((x, y, torch.ones_like(x))).to(bwd_rot)
#
#     # get world coords
#     world_coords = torch.matmul(bwd_rot, homog)
#     world_coords = world_coords * depth_map.reshape(1, -1)
#     world_coords = world_coords + bwd_trans.reshape(3, 1)
#     world_coords = torch.movedim(world_coords, 0, 1)
#     world_coords = world_coords.reshape(height, width, 3)
#
#     # get pixel projection
#     rot_coords = torch.matmul(rot, homog)
#     proj_3d = rot_coords * depth_map.reshape(1, -1)
#     proj_3d = proj_3d + trans.reshape(3, 1)
#     proj_2d = proj_3d[:2, :] / proj_3d[2:3, :]
#     proj_2d = (torch.movedim(proj_2d, 0, 1)).to(torch.long)
#     proj_2d = torch.flip(proj_2d, dims=(1,))
#
#     # compute projected depth
#     proj_depth = torch.sub(world_coords, C_ref).unsqueeze(-1)
#     proj_depth = torch.matmul(z_ref, proj_depth).reshape(height, width)
#     proj_depth = proj_depth.reshape(-1, 1)
#
#     # mask out invalid indices
#     mask = torch.where(proj_2d[:, 0] < height, 1, 0) * \
#            torch.where(proj_2d[:, 0] >= 0, 1, 0) * \
#            torch.where(proj_2d[:, 1] < width, 1, 0) * \
#            torch.where(proj_2d[:, 1] >= 0, 1, 0)
#     inds = torch.where(mask)[0]
#     proj_2d = torch.index_select(proj_2d, dim=0, index=inds)
#     proj_2d = (proj_2d[:, 0] * width) + proj_2d[:, 1]
#     proj_depth = torch.index_select(proj_depth, dim=0, index=inds).squeeze()
#
#     # find duplicate pixels, select the smallest depth
#     proj_2d_inverse, counts = proj_2d.unique(sorted=False,return_counts=True)
#     duplicates = proj_2d_inverse[counts>1]
#     for val in tqdm(duplicates):
#         min_depth = proj_depth[proj_2d==val].min()
#         proj_depth[proj_2d == val] = min_depth
#
#     warped_depth = torch.zeros(height * width).to(proj_2d.device)
#     warped_depth[proj_2d] = proj_depth
#     warped_depth = warped_depth.reshape(height, width)
#
#     return warped_depth