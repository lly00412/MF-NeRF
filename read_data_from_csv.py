import os
import pandas as pd
import argparse
def get_opts():
    parser = argparse.ArgumentParser()
    # data parameters
    parser.add_argument('--log_dir', type=str, required=True,
                        help='root directory of validation outputs')
    parser.add_argument('--scenes', type=str, default=None, nargs='+',
                        help='scens to evaluate')
    parser.add_argument('--file_name', type=str, default='eval_scores',
                        help='csv file name')
    parser.add_argument('--save_name', type=str, default='eval_summary',
                        help='csv file name to save summary')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_opts()

    evals = ['lpips_vgg', 'psnr', 'ssim']
    for scene in args.scenes:
        print(scene)
        result_df = pd.DataFrame(columns=['views','lpips','psnr','ssim'])
        log_path = os.path.join(args.log_dir, scene)

        vs_labels = os.listdir(log_path)
        if 'summary' in vs_labels:
            vs_labels.remove('summary')
        N_vs = len(vs_labels)
        for i in range(N_vs):
            log = f'{log_path}/{vs_labels[i]}/scores/{args.file_name}.csv'
            eval_data = pd.read_csv(log)
            result_df.at[i, 'views'] = int(vs_labels[i][2:])
            for key in eval_data.keys():
                value = eval_data.at[0,key]
                if isinstance(value,str):
                    if 'tensor' in value:
                        value = value.replace('(',',').replace(')',',').split(',')
                        value = float(value[1])
                result_df.at[i,key] = value

        os.makedirs(os.path.join(args.log_dir,scene,'summary'),exist_ok=True)
        file_path = f'{args.log_dir}/{scene}/summary/{args.save_name}.csv'
        print(f'Save to {file_path}')
        result_df.to_csv(file_path, index=False)

