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
#cuda = torch.device("cuda")

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

def Process_Data(instr, exp_dir, data_dir):
    dataset = h5py.File(os.path.join(data_dir, 'train_data.hdf5'),'r')
    score = dataset['{}_pianoroll'.format(instr)][:]
    spec = dataset['{}_spec'.format(instr)][:]
    onoff = dataset['{}_onoff'.format(instr)][:]
    score = np.concatenate((score, onoff),axis = -1)
    score = np.transpose(score,(0,2,1))

    X_train, X_test, Y_train, Y_test = train_test_split(score, spec, test_size=0.2) 
    
    test_data_dir = os.path.join(exp_dir,'test_data')
    os.makedirs(test_data_dir)
    
    np.save(os.path.join(test_data_dir, "test_X.npy"), X_test)
    np.save(os.path.join(test_data_dir, "test_Y.npy"), Y_test)    
    
    train_dataset = utils.TensorDataset(torch.Tensor(X_train), torch.Tensor(Y_train))
    train_loader = utils.DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_dataset = utils.TensorDataset(torch.Tensor(X_test), torch.Tensor(Y_test))
    test_loader = utils.DataLoader(test_dataset, batch_size=16, shuffle=True) 
    
    return train_loader, test_loader

def train(model, epoch, train_loader, optimizer,iter_train_loss):
    model.train()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):        
        optimizer.zero_grad()
        split = torch.split(data, 128, dim=1)
        y_pred = model(split[0],split[1])
        loss_function = nn.MSELoss()
        loss = loss_function(y_pred, target)
        loss.backward()
        iter_train_loss.append(loss.item())
        train_loss += loss
        optimizer.step()    
         
        if batch_idx % 2 == 0:
            print ('Train Epoch: {} [{}/{} ({:.0f}%)]\t Loss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx/len(train_loader), loss.item()/len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss/ len(train_loader.dataset)))
    return train_loss/ len(train_loader.dataset)

def test(model, epoch, test_loader, scheduler, iter_test_loss):
    with torch.no_grad():
        model.eval()
        test_loss = 0
        for idx, (data, target) in enumerate(test_loader):
            split = torch.split(data,128,dim = 1)
            y_pred = model(split[0],split[1])
            loss_function = nn.MSELoss() 
            loss = loss_function(y_pred,target)
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
    #model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    model.zero_grad()
    optimizer.zero_grad()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    train_loader, test_loader = Process_Data(hp.instrument, exp_dir, args.data_dir)
    print ('start training')
    for epoch in range(hp.train_epoch):
        loss = train(model, epoch, train_loader, optimizer,hp.iter_train_loss)
        hp.loss_history.append(loss.item())
        if epoch % hp.test_freq == 0:
            test_loss = test(model, epoch, test_loader, scheduler, hp.iter_test_loss)
            hp.test_loss_history.append(test_loss.item())
            if test_loss < hp.best_loss:         
                torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'optimizer' : optimizer.state_dict()}, os.path.join(exp_dir, 'checkpoint-{}.tar'.format(str(epoch + 1 ))))
                hp.best_loss = test_loss.item()    
                hp.best_epoch = epoch + 1    
                with open(os.path.join(exp_dir,'hyperparams.json'), 'w') as outfile:   
                    json.dump(hp.__dict__, outfile)
       

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-data-dir", type=str, help="directory where musicnet.npz is")
    parser.add_argument("-instrument", type=str)
    parser.add_argument("-epochs", type=int)
    parser.add_argument("-test-freq", type=int)
    parser.add_argument("-exp-name", type=str)
    args = parser.parse_args()
    
    main(args)
