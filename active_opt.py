import argparse

def get_opts():
    parser = argparse.ArgumentParser()
    #######################
    # dataset parameters
    #######################
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
    parser.add_argument("--init_vs", type=int, default=10,
                        help='size of initial trainset, if init_vs=0 means use the full trainset')
    parser.add_argument("--train_img", type=int, default=None, nargs='+',
                        help='only use training imgs listed here')
    parser.add_argument("--test_img", type=int, default=None, nargs='+',
                        help='only use test imgs listed here')
    parser.add_argument('--n_centers', type=int, default=0,
                        help='num of initial centers for nsvf training kmeans')

    #######################
    # experimental training options
    #######################
    parser.add_argument('--exp_name', type=str, default='exp',
                        help='experiment name')
    parser.add_argument('--optimize_ext', action='store_true', default=False,
                        help='whether to optimize extrinsics')
    parser.add_argument('--random_bg', action='store_true', default=False,
                        help='''whether to train with random bg color (real scene only)
                           to avoid objects with black color to be predicted as transparent
                           ''')

    #######################
    # model configs
    #######################
    parser.add_argument('--ckpt_path', type=str, default=None,
                        help='pretrained checkpoint to load (including optimizers, etc)')
    parser.add_argument('--weight_path', type=str, default=None,
                        help='pretrained checkpoint to load (excluding optimizers, etc)')
    parser.add_argument('--scale', type=float, default=0.5,
                        help='scene scale (whole scene must lie in [-scale, scale]^3')
    parser.add_argument('--use_exposure', action='store_true', default=False,
                        help='whether to train in HDR-NeRF setting')
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

    #######################
    # loss parameters
    #######################
    parser.add_argument('--distortion_loss_w', type=float, default=0,
                        help='''weight of distortion loss (see losses.py),
                        0 to disable (default), to enable,
                        a good value is 1e-3 for real scene and 1e-2 for synthetic scene
                        ''')
    parser.add_argument('--loss', type=str, default='l2',
                        choices=['l2', 'nll', 'nllc'],
                        help='which loss to train: l2, nagtive loglikihood, nagtive loglikelihood + consistency (by SEDNet)')

    #######################
    # training options
    #######################
    parser.add_argument('--batch_size', type=int, default=8192,
                        help='number of rays in a batch')
    parser.add_argument('--ray_sampling_strategy', type=str, default='all_images',
                        choices=['all_images', 'same_image','weighted_images'],
                        help='''
                        all_images: uniformly from all pixels of ALL images
                        same_image: uniformly from all pixels of a SAME image
                        more_new_images: select new images with higher probability and uniformly select pixels of each image
                        ''')
    parser.add_argument('--num_epochs', type=int, default=30,
                        help='number of training epochs')
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='number of gpus')
    parser.add_argument('--lr', type=float, default=1e-2,
                        help='learning rate')

    #######################
    # active learning view selection
    #######################
    parser.add_argument('--view_select', action='store_true', default=False,
                        help='whether run view selection process')
    parser.add_argument("--vs_seed", type=int, default=349,
                        help='random seed to initialize the training set')
    parser.add_argument("--pre_train_epoch", type=int, default=10,
                        help='num of pretrain epoch for the starting point')
    parser.add_argument("--N_vs", type=int, default=4,
                        help='run view selection process for N times')
    parser.add_argument("--select_k", type=int, default=5,
                        help='num of views add to trainset each time')
    parser.add_argument("--vs_step", type=int, default=5,
                        help='num of epochs between each selection')
    parser.add_argument('--vs_sample_rate', type=float, default=1.,
                        help='percentage of sampling rays per view, 1 means rendering all rays')
    parser.add_argument('--vs_batch_size', type=int, default=1024,
                        help='number of rays processing a batch for view selection')
    parser.add_argument('--vs_by', type=str, default=None,
                        choices=[None, 'random', 'warp', 'mcd_d', 'mcd_r', 'entropy','l2','grad'],
                        help='select supplemental views by random / warping uncertainty / mcdropout depth / mcdropout rgb')
    parser.add_argument('--no_save_vs', action='store_true', default=False,
                        help='whether to save vs uncertainty map')

    #######################
    # uncertainty option
    #######################
    parser.add_argument('--eval_u', action='store_true', default=False,
                        help='whether to compute uncertainty')
    parser.add_argument('--u_by', type=str, default=None, nargs='+',
                        choices=[None, 'warp', 'mcd_d', 'mcd_r','entropy','l2','grad'],
                        help='estimate uncertainty by warping / mcdropout depth / mcdropout rgb/ entropy')
    parser.add_argument('--plot_roc', action='store_true', default=False,
                        help='whether to plot roc of all estimation')

    # mcdropout settings
    parser.add_argument("--n_passes", type=int, default=30,
                        help='number of passes for mc_dropout')
    parser.add_argument("--p", type=float, default=0.2,
                        help='drop prob for mc_dropout')

    # vs-nerf settings
    parser.add_argument("--theta", type=int, default=1,
                        help='number of passes for mc_dropout')

    #######################
    # validation options
    #######################
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
    parser.add_argument('--save_csv', action='store_true', default=False,
                        help='save the evaluation results to csv')

    return parser.parse_args()
