import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from sklearn.model_selection import train_test_split
import torch.utils.data as utils
import h5py 
import sys
import os
import json
from model import PerformanceNet
import argparse
import os
import time

CUDA_FLAG = 0
if torch.cuda.is_available():
    cuda = torch.device("cuda")
    CUDA_FLAG = 1


class hyperparams(object):
    def __init__(self, args):
        self.instrument = args.instrument
        self.train_epoch = args.epochs
        self.test_freq = args.test_freq
        self.exp_name = args.exp_name
        self.iter_train_loss = []
        self.iter_test_loss = []
        self.loss_history = []
        self.test_loss_history = []
        self.best_loss = 1e10 
        self.best_epoch = 0


class Dataseth5py(torch.utils.data.Dataset):
    # https://discuss.pytorch.org/t/how-to-speed-up-the-data-loader/13740/3
    def __init__(self, in_file, instr, data_type='train', n_read=None):
        super(Dataseth5py, self).__init__()

        self.dataset = h5py.File(in_file, 'r')

        # TODO: the big issue is you need to optimize how to read data into memory from h5py
        # loading one-by-one is way too slow (a few seconds vs. microseconds). Note that the 
        # rest of the profiling times are: concat/transpose ~ 0.005s, FloatTensor ~ 0.02 
        # (and thus, after this loading issue is solved FloatTensor becomes the bottleneck unless it 
        # can be moved to the main train() function and be applied to batches vs individual items here)
        if data_type == 'train':
            index_i = 200
            index_f = 1000000 if n_read is None else (index_i + n_read)
        elif data_type == 'test':
            index_i = 0
            index_f = 200
        else:
            raise ValueError(f'data type {data_type} not recognized')
        
        self.score = self.dataset['{}_pianoroll'.format(instr)][index_i: index_f]
        self.spec = self.dataset['{}_spec'.format(instr)][index_i: index_f]
        self.onoff = self.dataset['{}_onoff'.format(instr)][index_i: index_f]
        self.n_data = self.spec.shape[0]

    def __getitem__(self, index):
        spec = self.spec[index]
        score = self.score[index]
        onoff = self.onoff[index]

        score = np.concatenate((score, onoff), axis = -1)
        score = np.transpose(score, (1, 0))

        if CUDA_FLAG == 1:
            X = torch.cuda.FloatTensor(score)
            y = torch.cuda.FloatTensor(spec)
        else:
            X = torch.Tensor(score)
            y = torch.Tensor(spec)
        return X, y

    def __len__(self):
        return self.n_data


def Process_Data(instr, exp_dir, data_dir, n_read=None, batch_size=16):
    train_dataset = Dataseth5py(os.path.join(data_dir, f'train_data_{instr}.hdf5'), instr, 'train', n_read)
    test_dataset = Dataseth5py(os.path.join(data_dir, f'train_data_{instr}.hdf5'), instr, 'test')

    kwargs = {}
    train_loader = utils.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = utils.DataLoader(test_dataset, batch_size=batch_size)
    return train_loader, test_loader


def train(model, epoch, train_loader, optimizer, iter_train_loss):
    model.train()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):        
        optimizer.zero_grad()
        split = torch.split(data, 128, dim=1)
        loss_function = nn.MSELoss()
        if CUDA_FLAG == 1:
            y_pred = model(split[0].cuda(), split[1].cuda())
            loss = loss_function(y_pred, target.cuda())
        else:
            y_pred = model(split[0], split[1]) 
            loss = loss_function(y_pred, target)
        
        loss.backward()
        iter_train_loss.append(loss.item())
        train_loss += loss
        optimizer.step()    
         
        if batch_idx % 2 == 0:
            print ('Train Epoch: {} [{}/{} ({:.0f}%)]\t Loss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx/len(train_loader), loss.item()/len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss/ len(train_loader.dataset)))
    return train_loss / len(train_loader.dataset)


def test(model, epoch, test_loader, scheduler, iter_test_loss):
    with torch.no_grad():
        model.eval()
        test_loss = 0
        for idx, (data, target) in enumerate(test_loader):
            split = torch.split(data, 128, dim=1)
            loss_function = nn.MSELoss()
            if CUDA_FLAG == 1:
                y_pred = model(split[0].cuda(), split[1].cuda())
                loss = loss_function(y_pred, target.cuda())
            else:
                y_pred = model(split[0], split[1])
                loss = loss_function(y_pred, target)
            iter_test_loss.append(loss.item())
            test_loss += loss    
        test_loss/= len(test_loader.dataset)
        scheduler.step(test_loss)
        print ('====> Test set loss: {:.4f}'.format(test_loss))
        return test_loss


def main(args):
    hp = hyperparams(args)

    try:
        exp_root = os.path.join(os.path.abspath('./'),'experiments')
        os.makedirs(exp_root)
    except FileExistsError:
        pass
    
    exp_dir = os.path.join(exp_root, hp.exp_name)
    os.makedirs(exp_dir)

    model = PerformanceNet()
    if CUDA_FLAG == 1:
        model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    model.zero_grad()
    optimizer.zero_grad()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    train_loader, test_loader = Process_Data(hp.instrument, exp_dir, args.data_dir, args.n_read, args.batch_size)
    print ('start training')
    for epoch in range(hp.train_epoch):
        loss = train(model, epoch, train_loader, optimizer, hp.iter_train_loss)
        hp.loss_history.append(loss.item())
        
        # test
        if epoch % hp.test_freq == 0:
            test_loss = test(model, epoch, test_loader, scheduler, hp.iter_test_loss)
            hp.test_loss_history.append(test_loss.item())
            if test_loss < hp.best_loss:
                print("saving model")         
                torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'optimizer' : optimizer.state_dict()}, os.path.join(exp_dir, 'checkpoint-{}.tar'.format(str(epoch + 1 ))))
                hp.best_loss = test_loss.item()    
                hp.best_epoch = epoch + 1    
                with open(os.path.join(exp_dir, 'hyperparams.json'), 'w') as outfile:   
                    json.dump(hp.__dict__, outfile)
       

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-data-dir", type=str, default='/Users/arisilburt/Machine_Learning/music/PerformanceNet_ari/data/', help="directory where musicnet.npz is")
    parser.add_argument("-instrument", type=str, default='cello')
    parser.add_argument("-epochs", type=int, default=1)
    parser.add_argument("-test-freq", type=int, default=1)
    parser.add_argument("-exp-name", type=str, default='cello_test')
    parser.add_argument("--n-read", type=int, default=None, help='How many data points to read (length of an epoch)')
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    
    main(args)
