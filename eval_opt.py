import argparse

def get_opts():
    parser = argparse.ArgumentParser()

    # data parameters
    parser.add_argument('--val_dir', type=str, required=True,
                        help='root directory of validation outputs')
    parser.add_argument('--scenes', type=str, default=None, nargs='+',
                        help='scens to evaluate')

    # compute auc
    parser.add_argument('--opt', type=str, default='err',
                        help='optimums to compute auc')
    parser.add_argument('--est', type=str, default='u_pred', nargs='+',
                        help='options to estimate auc')
    parser.add_argument('--intervals', type=int, default=20,
                        help='num of roc points to compute auc')
    parser.add_argument('--plot_roc', action='store_true', default=False,
                        help='if plot roc curves')
    parser.add_argument('--plot_metric', action='store_true', default=False,
                        help='if plot per-pixel metric')

    # log
    parser.add_argument('--log_dir', type=str, default=None,
                        help='root directory of evaluation logs')
    parser.add_argument('--log_file', type=str, default='evaluation.txt',
                        help='name of evaluation logs')

    return parser.parse_args()
