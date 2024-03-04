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
    parser.add_argument('--method', type=str, default=None,
                        help='method name')
    parser.add_argument('--N_vs', type=int, default=1,
                        help='how many steps')
    parser.add_argument('--version', type=int, default=0,
                        help='which run version')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_opts()
    for scene in args.scenes:
        print(scene)
        if not args.method==None:
            log_path = os.path.join(args.log_dir,scene,args.method,f'*/version_{args.version}/events.*')
            logs = sorted(glob.glob(log_path))
        else:
            log_path = os.path.join(args.log_dir, scene,f'version_{args.version}/events.*')
            logs = sorted(glob.glob(log_path))
        result_df = pd.DataFrame()
        labels = ['22','24','26','28']
        evals = ['lpips_vgg','psnr','ssim']
        N_vs = args.N_vs
        for i in range(0, args.N_vs):
            acc = EventAccumulator(logs[i])
            acc.Reload()
            for eval in evals:
                df = pd.DataFrame(acc.Scalars(f"test/{eval}"))
                result_df.at[i,eval] = df['value'].iloc[-1]
        if not args.method == None:
            file_path = os.path.join(args.log_dir,scene,args.method,'tensorboard_data.csv')
        else:
            file_path = os.path.join(args.log_dir, scene,'tensorboard_data.csv')
        print(f'Save to {file_path}')
        result_df.to_csv(file_path, index=False)

    # N_vs = args.N_vs
    # N_scenes = len(args.scenes)
    # labels = ['22', '24', '26', '28']
    # evals = ['train/runtime(mins)', 'vs/runtime(mins)']
    # result_df = pd.DataFrame()
    # for i in range(1,args.N_vs):
    #     for eval in evals:
    #         acc_values = 0
    #         for scene in args.scenes:
    #             if not args.method==None:
    #                 log_path = os.path.join(args.log_dir,scene,args.method,'*/version_*/events.*')
    #                 logs = sorted(glob.glob(log_path))
    #             else:
    #                 log_path = os.path.join(args.log_dir, scene,'version_*/events.*')
    #                 logs = sorted(glob.glob(log_path))
    #             acc = EventAccumulator(logs[i])
    #             acc.Reload()
    #             df = pd.DataFrame(acc.Scalars(eval))
    #             acc_values += df['value'].iloc[-1]
    #         result_df.at[i,eval] = acc_values/N_scenes
    # if not args.method == None:
    #     os.makedirs(os.path.join(args.log_dir,args.method), exist_ok=True)
    #     file_path = os.path.join(args.log_dir,args.method,'avg_runtime.csv')
    # else:
    #     file_path = os.path.join(args.log_dir, 'avg_runtime.csv')
    # print(f'Save to {file_path}')
    # result_df.to_csv(file_path, index=False)

