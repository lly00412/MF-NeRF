import torch
from torch import nn
# data
from torch.utils.data import TensorDataset,DataLoader

from utils import *
import time

import warnings; warnings.filterwarnings("ignore")
import pandas as pd
import argparse
import random

def str2float(strlist):
    strlist = strlist[1:-1].split(',')
    return [float(x.strip()) for x in strlist]
def create_dataset_across_scenes(data_file,test_scene='Drums',target='psnr'):
    # read in and create data
    raw_df = pd.read_csv(data_file)
    test_df = raw_df[raw_df['scene'] == test_scene] # only test on one scene!
    train_df = raw_df[raw_df['scene'] != test_scene]

    # train dataset
    x_train = []
    y_train = []
    for scene in train_df['scene'].unique():
        s_df = train_df[train_df['scene'] == scene]
        n_views = len(s_df)
        for i in range(n_views):
            for j in range(n_views):
                if not i==j:
                    f1 = str2float(s_df['u_hist'].iloc[i])
                    f2 = str2float(s_df['u_hist'].iloc[j])
                    f3 = [a-b for a,b in zip(f1,f2)]
                    # f4 = s_df['avg_sigma'].iloc[i] - s_df['avg_sigma'].iloc[j]
                    # f3.append(f4)
                    label = float(s_df[target].iloc[i]>s_df[target].iloc[j])
                    # x_train.append([f1,f2,f3])
                    x_train.append(f3)
                    y_train.append(label)

    x_train = torch.tensor(x_train,dtype=torch.float32)
    y_train = torch.tensor(y_train,dtype=torch.float32).unsqueeze(1)
    train_dataset = TensorDataset(x_train, y_train)

    # test dataset
    x_test = []
    y_test = []
    test_views = len(test_df)
    for i in range(test_views):
        for j in range(test_views):
            if not i == j:
                f1 = str2float(test_df['u_hist'].iloc[i])
                f2 = str2float(test_df['u_hist'].iloc[j])
                f3 = [a - b for a, b in zip(f1, f2)]
                # f4 = test_df['avg_sigma'].iloc[i] - test_df['avg_sigma'].iloc[j]
                # f3.append(f4)
                label = int(test_df[target].iloc[i] > test_df[target].iloc[j])
                # x_test.append([f1, f2, f3])
                x_test.append(f3)
                y_test.append(label)

    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
    test_dataset = TensorDataset(x_test, y_test)

    return train_dataset,test_dataset

def create_dataset_on_same_scene(data_file,test_scene='Drums',target='psnr'):
    # read in and create data
    raw_df = pd.read_csv(data_file)
    scene_df = raw_df[raw_df['scene'] == test_scene] # only test on one scene!

    # create dataset
    X = []
    Y = []
    n_views = len(scene_df)
    for i in range(n_views):
        for j in range(n_views):
            if not i==j:
                f1 = str2float(scene_df['u_hist'].iloc[i])
                f2 = str2float(scene_df['u_hist'].iloc[j])
                f3 = [a-b for a,b in zip(f1,f2)]
                # f4 = s_df['avg_sigma'].iloc[i] - s_df['avg_sigma'].iloc[j]
                # f3.append(f4)
                label = float(scene_df[target].iloc[i] > scene_df[target].iloc[j])
                if target == 'lpips':
                    label = 1 - label

                # x_train.append([f1,f2,f3])
                X.append(f3)
                Y.append(label)

    # shuffle
    random_seed = 48097
    random.seed(random_seed)
    combined_lists = list(zip(X, Y))
    random.shuffle(combined_lists)
    shuffle_X, shuffle_Y = zip(*combined_lists)
    shuffle_X = torch.tensor(shuffle_X, dtype=torch.float32)
    shuffle_Y = torch.tensor(shuffle_Y, dtype=torch.float32).unsqueeze(1)

    # split
    split_rate = 0.8
    n_train = int(len(shuffle_Y)*0.8)

    x_train,y_train = shuffle_X[:n_train],shuffle_Y[:n_train]
    x_test,y_test = shuffle_X[n_train:],shuffle_Y[n_train:]

    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)

    return train_dataset,test_dataset
def get_opts():
    parser = argparse.ArgumentParser()

    # dataset parameters
    parser.add_argument('--data_file', type=str, required=True,
                        help='csv file to create dataset')
    parser.add_argument('--dataset_name', type=str, default='nerfvs',
                        choices=['nerfvs', 'nsvfvs', 'llffvs'],
                        help='which dataset to train/test')
    parser.add_argument('--test_scene', type=str, required=True,default='Drums',
                        choices=['Hotdog','Chair','Ficus','Drums'],
                        help='test on which scene')
    parser.add_argument('--target', type=str, default='psnr',
                        choices=['psnr','ssim','lpips'],
                        help='train the classifier based on which metric')
    parser.add_argument("--data_seed", type=int, default=34958,
                        help='random seed to initialize the training set')

    # training options
    parser.add_argument('--batch_size', type=int, default=128,
                        help='number of samples in a batch')
    parser.add_argument('--num_epochs', type=int, default=20,
                        help='number of training epochs')
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='number of gpus')
    parser.add_argument('--lr', type=float, default=1e-2,
                        help='learning rate')

    # loss options
    parser.add_argument('--loss', type=str, default='l2',
                        choices=['bce', 'nll'],
                        help='which loss to train')

    # validation options
    parser.add_argument('--val_only', action='store_true', default=False,
                        help='run only validation (need to provide ckpt_path)')

    # misc
    parser.add_argument('--exp_name', type=str, default='exp',
                        help='experiment name')
    parser.add_argument('--ckpt_path', type=str, default=None,
                        help='pretrained checkpoint to load (including optimizers, etc)')

    parser.add_argument('--seed', type=int, default=1337,
                        help='random seed')

    return parser.parse_args()

class BinarryClassifier(nn.Module):
    def __init__(self, indim=3*10, n_classes=2,act='Sigmoid'):
        super().__init__()
        if act=='Sigmoid':
            self.act = nn.Sigmoid()
        if act=='Softmax':
            self.act = nn.Softmax()
        if act == 'logSoftmax':
            self.act = nn.LogSoftmax()

        self.fc1 = nn.Linear(indim, 128)  # Fully connected layer 1
        self.relu = nn.ReLU()  # ReLU activation
        self.fc2 = nn.Linear(128, 128)  # Fully connected layer 2
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, n_classes)
        # self.fc = nn.Linear(indim,n_classes)
        self.dropout = nn.Dropout(p=0.2)
    def forward(self,x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x= self.fc4(x)
        x = self.relu(x)
        x = self.act(x)
        return x


def train(hparams):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model = BinarryClassifier(indim=11, n_classes=1,act='Sigmoid')
    epochs = hparams.num_epochs
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.to(device)

    train_dataset, test_dataset = create_dataset_on_same_scene(data_file=hparams.data_file,
                                                           test_scene=hparams.test_scene,
                                                           target=hparams.target)

    train_dataloader = DataLoader(train_dataset,
                          num_workers=16,
                          persistent_workers=True,
                          batch_size=hparams.batch_size,
                          pin_memory=True,
                          shuffle=True)
    val_dataloader = DataLoader(test_dataset,
                          num_workers=8,
                          batch_size=8,
                          pin_memory=True)

    for epoch in range(epochs):
        total_loss = 0.0
        corrects = 0.0
        total_samples = 0.0

        model.train()

        for inputs, targets in train_dataloader:
            inputs,targets = inputs.to(device),targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            corrects += (outputs.round()  == targets).sum().item()
            total_samples += len(targets)

        average_loss = total_loss / len(train_dataloader)
        accuracy = corrects / total_samples
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {average_loss:.4f}, Accuracy: {accuracy*100:.2f}%')

    os.makedirs(f'ckpts/{hparams.dataset_name}/{hparams.exp_name}/', exist_ok=True)
    results_file = f'ckpts/{hparams.dataset_name}/{hparams.exp_name}/{hparams.num_epochs:d}.txt'
    with open(results_file, 'a') as f:
        f.write(f'Scene: {hparams.test_scene}, Metric: {hparams.target} \n')
        f.write(f'Epoch [{epoch + 1}/{epochs}], Loss: {average_loss:.4f}, Accuracy: {accuracy * 100:.2f}% \n')
        f.close()

    model.eval()
    corrects = 0.0
    total_samples = 0.0
    true_positives = 0.0
    total_positives = 0.0

    for inputs, targets in val_dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        corrects += (outputs.round() == targets).sum().item()
        total_samples += len(targets)

        true_positives += ((outputs.round() == 1) & (targets == 1)).sum()
        total_positives += (targets == 1).sum()

    accuracy = corrects / total_samples
    precision = true_positives / (total_positives + 1e-10)  # to avoid division by zero
    recall = true_positives / (total_samples + 1e-10)  # to avoid division by zero
    f1 = 2 * (precision * recall) / (precision + recall + 1e-10)  #

    print(f'Test evaluation, Accuracy: {accuracy * 100:.2f}%, F1: {f1:.4f}')

    torch.save(model.state_dict(), f'ckpts/{hparams.dataset_name}/{hparams.exp_name}/{hparams.num_epochs:d}.pth')
    with open(results_file, 'a') as f:
        f.write(f'Test evaluation, Accuracy: {accuracy * 100:.2f}%, F1: {f1:.4f} \n')
        f.close()

if __name__ == '__main__':
    start = time.time()
    hparams = get_opts()

    torch.manual_seed(hparams.seed)

    print(f'Train on scene: {hparams.test_scene}')
    train(hparams)

    end = time.time()
    runtime = time.strftime("%H:%M:%S", time.gmtime(end - start))
    print('Total runtime: {}'.format(runtime))


