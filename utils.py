import torch
import matplotlib.pyplot as plt
import numpy as np
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

def extract_model_state_dict(ckpt_path, model_name='model', prefixes_to_ignore=[]):
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    checkpoint_ = {}
    if 'state_dict' in checkpoint: # if it's a pytorch-lightning checkpoint
        checkpoint = checkpoint['state_dict']
    for k, v in checkpoint.items():
        if not k.startswith(model_name):
            continue
        k = k[len(model_name)+1:]
        for prefix in prefixes_to_ignore:
            if k.startswith(prefix):
                break
        else:
            checkpoint_[k] = v
    return checkpoint_


def load_ckpt(model, ckpt_path, model_name='model', prefixes_to_ignore=[]):
    if not ckpt_path: return
    model_dict = model.state_dict()
    checkpoint_ = extract_model_state_dict(ckpt_path, model_name, prefixes_to_ignore)
    model_dict.update(checkpoint_)
    model.load_state_dict(model_dict)


def slim_ckpt(ckpt_path, save_poses=False):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    # pop unused parameters
    keys_to_pop = ['directions', 'model.density_grid', 'model.grid_coords']
    if not save_poses: keys_to_pop += ['poses']
    for k in ckpt['state_dict']:
        if k.startswith('val_lpips'):
            keys_to_pop += [k]
    for k in keys_to_pop:
        ckpt['state_dict'].pop(k, None)
    return ckpt['state_dict']

def warp_tgt_to_ref(tgt_depth, ref_cams, tgt_cams, device='cpu'):
    # warp tgt depth map to ref view
    height, width = tgt_depth.shape
    # grab intrinsics and extrinsics from reference view
    P_ref = torch.zeros(4,4)
    P_ref[:3,:] = ref_cams['pose'] # 3x4 -> 4x4
    P_ref[-1,-1] = 1

    K_ref = torch.zeros(4,4)
    K_ref[:3,:3] = ref_cams['K'] # 3x3 -> 4x4
    K_ref[-1, -1] = 1

    P_ref = P_ref.to(device)
    K_ref = K_ref.to(device)

    R_ref = P_ref[:3, :3] # 3x3
    t_ref = P_ref[:3, 3:4] # 3x1

    C_ref = torch.matmul(-R_ref.transpose(0, 1), t_ref)
    z_ref = R_ref[2:3, :3].reshape(1, 1, 1, 3).repeat(height, width, 1, 1)
    C_ref = C_ref.reshape(1, 1, 3).repeat(height, width, 1)

    depth_map = tgt_depth.to(device)  # h,w

    # get intrinsics and extrinsics from target view
    P_tgt = torch.zeros(4, 4)
    P_tgt[:3, :] = tgt_cams['pose']  # 3x4 -> 4x4
    P_tgt[-1, -1] = 1

    K_tgt = torch.zeros(4, 4)
    K_tgt[:3, :3] = tgt_cams['K']  # 3x3 -> 4x4
    K_tgt[-1, -1] = 1

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

    warped_depth = torch.zeros(height * width).to(proj_2d.device)
    warped_depth[proj_2d] = proj_depth
    warped_depth = warped_depth.reshape(height, width)

    return warped_depth


def percentile(t: torch.tensor, q):
    """
    Return the ``q``-th percentile of the flattened input tensor's data.

    CAUTION:
     * Needs PyTorch >= 1.1.0, as ``torch.kthvalue()`` is used.
     * Values are not interpolated, which corresponds to
       ``numpy.percentile(..., interpolation="nearest")``.

    :param t: Input tensor.
    :param q: Percentile to compute, which must be between 0 and 100 inclusive.
    :return: Resulting value (scalar).
    """
    # Note that ``kthvalue()`` works one-based, i.e. the first sorted value
    # indeed corresponds to k=1, not k=0! Use float(q) instead of q directly,
    # so that ``round()`` returns an integer, even if q is a np.float32.
    k = 1 + round(.01 * float(q) * (t.numel() - 1))
    result = t.view(-1).kthvalue(k).values.item()
    return result

def compute_roc(opt,est,intervals = 20):
    ROC = []
    quants = [100. / intervals * t for t in range(1, intervals + 1)]
    thres = [np.percentile(est, q) for q in quants]
    subs = [est <= t for t in thres]
    ROC_points = [opt[s].mean() for s in subs]
    [ROC.append(r) for r in ROC_points]
    AUC = np.trapz(ROC, dx=1. / intervals)
    return ROC,AUC

def plot_roc(ROC_opt,ROC_est,fig_name, intervals = 20, opt_label='rgb_err',est_label='depth_err'):
    quants = [100. / intervals * t for t in range(1, intervals + 1)]
    plt.figure()
    plt.plot(quants, ROC_opt, marker="^", markersize=8,  color='blue', label=opt_label)
    plt.plot(quants, ROC_est, marker="o", markersize=8,  color='red', label=est_label)
    plt.xticks(quants)
    plt.xlabel('Sample Size(%)')
    plt.ylabel('Accumulative MSE')
    plt.legend()
    fig = plt.gcf()
    fig.set_size_inches(20, 8)
    plt.rcParams.update({'font.size': 30})
    fig.savefig(fig_name)
    plt.close()
