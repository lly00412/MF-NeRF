import os
import glob
import shutil

if __name__ == '__main__':
    dataset_dir = '/mnt/Data2/datasets/nerf_llff_data/'
    scenes=['fern','flower','fortress', 'horns', 'leaves','orchids','room','trex']
    for scene in scenes:
        img_root = os.path.join(dataset_dir,scene,'images')
        img_names = sorted(glob.glob(os.path.join(img_root, '*.???')))
        for factor in [4,8]:
            img_downdir = img_root+f'_{factor}'
            imgs_dows = sorted(glob.glob(os.path.join(img_downdir, '*.???')))
            for old_name, img_name in zip(imgs_dows,img_names):
                new_name = os.path.join(img_downdir,os.path.split(img_name)[-1])
                shutil.copy(old_name, new_name)