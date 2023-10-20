import torch
import cv2
import argparse
import os
import numpy as np
def Project2d(K,c2w,pt3d):
    K = torch.cat([K,torch.tensor([[0],[0],[0]])],dim=1) # 3x4
    bottom = torch.tensor([[0, 0, 0, 1.]])
    c2w = torch.cat([c2w, bottom]).to(torch.float32) # 4x4
    w2c = np.linalg.inv(c2w.numpy())
    w2c = torch.from_numpy(w2c)
    P = torch.matmul(K,w2c) # 3x4
    N_pts,_ = pt3d.shape
    pt3d_homo = torch.cat([pt3d,torch.ones(N_pts).reshape(-1,1)],dim=1).to(torch.float32)  # (N,3) -> (N,4)
    pt2d = torch.matmul(P,pt3d_homo.T) # pt3d should be (4,1)
    pt2d = pt2d.T
    pt2d[...,0] /= pt2d[...,2]
    pt2d[...,1] /= pt2d[...,2]
    return pt2d[...,:2]

def get_opts():
    parser = argparse.ArgumentParser()

    # data parameters
    parser.add_argument('--val_dir', type=str, required=True,
                        help='root directory of validation outputs')
    parser.add_argument('--scene', type=str, default=None,
                        help='scens to evaluate')

    parser.add_argument('--random_seed', type=int, default=999,
                        help='ramdom seed to generate test points')
    parser.add_argument('--N_points', type=int, default=10,
                        help='num of points for test')

    # log
    parser.add_argument('--log_dir', type=str, default=None,
                        help='root directory of evaluation logs')
    parser.add_argument('--log_file', type=str, default='evaluation.txt',
                        help='name of evaluation logs')

    return parser.parse_args()

def extract_data(hparams):
    cams_f = os.path.join(hparams.val_dir, hparams.scene, 'cams.pth')
    cams = torch.load(cams_f,map_location='cpu')
    pts_f = os.path.join(hparams.val_dir, hparams.scene, 'pts3d.pth')
    pts3d = torch.load(pts_f,map_location='cpu')

    pts3d_xyz = []
    for pt in pts3d['center']:
        pt_xyz = pt[:3] / pt[3]
        pts3d_xyz += [pt_xyz]
    pts3d_xyz = np.stack(pts3d_xyz)
    pts3d['center'] = pts3d_xyz
    return cams,pts3d

def generate_test_pts(hparams,pts3d):
    np.random.seed(hparams.random_seed)
    N_pts, _ = pts3d['raw'].shape
    idxs = np.random.randint(N_pts,size=hparams.N_points)
    pts3d, pts3d_center = pts3d['raw'][idxs], pts3d['center'][idxs]
    return pts3d,pts3d_center

if __name__ == '__main__':
    hparams = get_opts()
    cams, pts3d = extract_data(hparams) # cams: tensor, pts3d: numpy
    samle_pts3d,sample_pts3d_center = generate_test_pts(hparams,pts3d)
    sample_pts3d = torch.from_numpy(samle_pts3d)
    sample_pts3d_center = torch.from_numpy(sample_pts3d_center)
    scale = cams['scale']
    N_cams,_,_ = cams['raw_poses'].shape
    raw_pose = cams['raw_poses'][0]
    pose = cams['poses'][0]
    pts2d = Project2d(cams['K'], raw_pose, sample_pts3d)
    pts2d_center = Project2d(cams['K'], pose, sample_pts3d_center)
    for pt in range(hparams.N_points):
        print(f'points 3d: {sample_pts3d[pt]}')
        print(f'points 3d centered: {sample_pts3d_center[pt]}')
        print(f'points 2d : {pts2d[pt]}')
        print(f'points 2d centered: {pts2d_center[pt]}')




