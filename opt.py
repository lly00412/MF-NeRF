import argparse

def get_opts():
    parser = argparse.ArgumentParser()

    # dataset parameters
    parser.add_argument('--root_dir', type=str, required=True,
                        help='root directory of dataset')
    parser.add_argument('--dataset_name', type=str, default='nsvf',
                        choices=['nerf', 'nsvf', 'colmap', 'nerfpp', 'rtmv'],
                        help='which dataset to train/test')
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'trainval', 'trainvaltest'],
                        help='use which split to train')
    parser.add_argument('--downsample', type=float, default=1.0,
                        help='downsample factor (<=1.0) for the images')

    # model parameters
    parser.add_argument('--scale', type=float, default=0.5,
                        help='scene scale (whole scene must lie in [-scale, scale]^3')
    parser.add_argument('--use_exposure', action='store_true', default=False,
                        help='whether to train in HDR-NeRF setting')

    # loss parameters
    parser.add_argument('--distortion_loss_w', type=float, default=0,
                        help='''weight of distortion loss (see losses.py),
                        0 to disable (default), to enable,
                        a good value is 1e-3 for real scene and 1e-2 for synthetic scene
                        ''')

    # training options
    parser.add_argument('--batch_size', type=int, default=8192,
                        help='number of rays in a batch')
    parser.add_argument('--ray_sampling_strategy', type=str, default='all_images',
                        choices=['all_images', 'same_image'],
                        help='''
                        all_images: uniformly from all pixels of ALL images
                        same_image: uniformly from all pixels of a SAME image
                        ''')
    parser.add_argument('--num_epochs', type=int, default=30,
                        help='number of training epochs')
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='number of gpus')
    parser.add_argument('--lr', type=float, default=1e-2,
                        help='learning rate')

    # experimental training options
    parser.add_argument('--optimize_ext', action='store_true', default=False,
                        help='whether to optimize extrinsics')
    parser.add_argument('--random_bg', action='store_true', default=False,
                        help='''whether to train with random bg color (real scene only)
                        to avoid objects with black color to be predicted as transparent
                        ''')
    parser.add_argument("--fewshot_seed", type=int, default=340,
                        help='fewshot_seed')
    parser.add_argument("--fewshot", type=int, default=0,
                        help='if 0 not using fewshot, else: using fewshot')

    # loss options
    parser.add_argument('--loss', type=str, default='l2',
                        choices=['l2', 'kg', 'uc'],
                        help='which loss to train')
    parser.add_argument('--uncert', type=bool, default=False,
                        help='whether to estimate uncertainty')

    # validation options
    parser.add_argument('--eval_lpips', action='store_true', default=False,
                        help='evaluate lpips metric (consumes more VRAM)')
    parser.add_argument('--val_only', action='store_true', default=False,
                        help='run only validation (need to provide ckpt_path)')
    parser.add_argument('--no_save_test', action='store_true', default=False,
                        help='whether to save test image and video')
    parser.add_argument('--save_output', action='store_true', default=False,
                        help='save the raw outputs')
    parser.add_argument('--save_video', action='store_true', default=False,
                        help='save the render video')

    # mcdropout
    parser.add_argument("--mcdropout", action='store_true',
                        help='if do mc_dropout')
    parser.add_argument("--n_passes", type=int, default=10,
                        help='number of passes for mc_dropout')
    parser.add_argument("--p", type=float, default=0.5,
                        help='drop prob for mc_dropout')

    # misc
    parser.add_argument('--exp_name', type=str, default='exp',
                        help='experiment name')
    parser.add_argument('--ckpt_path', type=str, default=None,
                        help='pretrained checkpoint to load (including optimizers, etc)')
    parser.add_argument('--weight_path', type=str, default=None,
                        help='pretrained checkpoint to load (excluding optimizers, etc)')

    # network config
    parser.add_argument('--grid', type=str, default='Hash',
                        choices=['Hash', 'Window', "MixedFeature"],
                        help='Encoding scheme Hash or MixedFeature')
    parser.add_argument('--L', type=int, default=16,
                        help='Encoding hyper parameter L')
    parser.add_argument('--F', type=int, default=2,
                        help='Encoding hyper parameter F')
    parser.add_argument('--T', type=int, default=19,
                        help='Encoding hyper parameter T')
    parser.add_argument('--N_min', type=int, default=16,
                        help='Encoding hyper parameter N_min')
    parser.add_argument('--N_max', type=int, default=2048,
                        help='Encoding hyper parameter N_max')
    parser.add_argument('--N_tables', type=int, default=1,
                        help='Number of hash tables')

    parser.add_argument('--rgb_channels', type=int, default=64,
                        help='rgb network channels')
    parser.add_argument('--rgb_layers', type=int, default=2,
                        help='rgb network layers')

    parser.add_argument('--seed', type=int, default=1337,
                        help='random seed')

    return parser.parse_args()
