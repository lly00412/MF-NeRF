import os
import glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd
import argparse
def get_opts():
    parser = argparse.ArgumentParser()
    # data parameters
    parser.add_argument('--log_dir', type=str, required=True,
                        help='root directory of validation outputs')
    parser.add_argument('--scenes', type=str, default=None, nargs='+',
                        help='scens to evaluate')
    parser.add_argument('--eval_u', action='store_true', default=False,
                        help='whether to compute uncertainty')
    parser.add_argument('--u_by', type=str, default=None, nargs='+',
                        choices=[None, 'warp', 'mcd_d', 'mcd_r', 'entropy'])
    parser.add_argument('--N_vs', type=int, default=1,
                        help='how many models')
    parser.add_argument('--v_num', type=int, default=0,
                        help='version number')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_opts()

    evals = ['lpips_vgg', 'psnr', 'ssim']
    for u_method in args.u_by:
        evals.append(f'u_{u_method}')

    for scene in args.scenes:
        print(scene)
        result_df = pd.DataFrame()
        log_path = os.path.join(args.log_dir, scene)

        if args.N_vs > 1:
            vs_labels = os.listdir(log_path)
        else:
            vs_labels = ['']

        for i in range(args.N_vs):
            log = glob.glob(f'{log_path}/{vs_labels[i]}/version_{args.v_num}/events.*')
            # print(log)
            acc = EventAccumulator(log[-1])
            acc.Reload()
            result_df.at[i, 'views'] = vs_labels[i]
            for eval in evals:
                df = pd.DataFrame(acc.Scalars(f"test/{eval}"))
                result_df.at[i,eval] = df['value'].iloc[-1]

        file_path = os.path.join(args.log_dir,scene,'eval_summary.csv')
        print(f'Save to {file_path}')
        result_df.to_csv(file_path, index=False)

