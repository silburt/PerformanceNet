import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from sklearn.model_selection import train_test_split
import torch.utils.data as utils
import librosa
import h5py 
import sys
import os
import json
from model import PerformanceNet
cuda = torch.device("cuda")

from process_data import hyperparams as pp_hyperparams
import random
random.seed(42)

pp_hp = pp_hyperparams()

class hyperparams(object):
    def __init__(self):
        self.instrument = sys.argv[1]
        self.train_epoch = int(sys.argv[2]) #default = 300
        self.test_freq = int(sys.argv[3])  #default = 10 
        self.exp_name = sys.argv[4]
        self.iter_train_loss = []
        self.iter_test_loss = []
        self.loss_history = []
        self.test_loss_history = []
        self.best_loss = 1e10 
        self.best_epoch = 0


TEST_TRAIN_SPLIT_PERCENTAGE = 0.2

class DatasetPreprocessRealTime(torch.utils.data.Dataset):
    def __init__(self, in_file, instr, exp_dir, data_type='train'):
        super(DatasetPreprocessRealTime, self).__init__()

        self.dataset = h5py.File(in_file, 'r')

        # load all the raw audio files
        self.audios = {key: self.dataset[key] for key in self.dataset.keys() if 'audio_' in key and instr in key}

        self.pianoroll = self.dataset[instr + "_pianoroll"][:]
        self.onoff = self.dataset[instr + "_onoff"][:]
        self.target_coords = self.dataset[instr + "_spec_coords"][:]

        # test/train split
        indices = list(range(self.pianoroll.shape[0]))
        random.shuffle(indices)
        split_index = int(len(indices) * TEST_TRAIN_SPLIT_PERCENTAGE)
        valid_indices = indices[:split_index] if data_type == 'test' else indices[split_index:]

        # reduce data to test/train
        self.pianoroll = self.pianoroll[valid_indices]
        self.onoff = self.onoff[valid_indices]
        self.target_coords = self.target_coords[valid_indices]

        self.n_data = self.pianoroll.shape[0]
        self.instr = instr
        self.exp_dir = exp_dir


    def write_test_data(self):
        test_data_dir = os.path.join(self.exp_dir, 'test_data')
        os.makedirs(test_data_dir)
        
        print(f"Writing test data to {test_data_dir}")
        score = np.concatenate((self.pianoroll, self.onoff), axis = -1)
        X_test = np.transpose(score, (0,2,1))

        specs = []
        for index in range(self.n_data):
            _, _, audio_chunk = self.select_piano_and_audio_chunks(index)
            spec = librosa.stft(audio_chunk, n_fft= pp_hp.n_fft, hop_length = pp_hp.stride)
            y = np.log1p(np.abs(spec)**2)
            specs.append(y)
        Y_test = np.array(specs)
        
        np.save(os.path.join(test_data_dir, "test_X.npy"), X_test)
        np.save(os.path.join(test_data_dir, "test_Y.npy"), Y_test)
        print("Successfully wrote test data")


    def select_piano_and_audio_chunks(self, index):
       # piano
        pianoroll = self.pianoroll[index]
        onoff = self.onoff[index]

        # get correct audio_chunk of selected style for output target
        song_id, chunk_begin_index, chunk_end_index = self.target_coords[index].astype('int')
        audio_chunk = self.audios[self.instr + "_audio_" + str(song_id)][chunk_begin_index: chunk_end_index]

        return pianoroll, onoff, audio_chunk


    def __getitem__(self, index):
        '''
        The input data are the pianoroll, onoff, target_coords
        The spectrogram is calculated on-the-fly (to save space) for the corresponding pianoroll/onoff
        '''
        score, onoff, audio_chunk = self.select_piano_and_audio_chunks(index)

        # prepare pianoroll
        #score = np.concatenate((score, onoff),axis = -1)
        #score = np.transpose(score,(0,2,1))
        pianoroll = np.concatenate((pianoroll, onoff), axis=-1)
        pianoroll = np.transpose(pianoroll, (1, 0))

        # target spectrogram
        spec = librosa.stft(audio_chunk, n_fft= pp_hp.n_fft, hop_length = pp_hp.stride)
        y = np.log1p(np.abs(spec)**2)

        # cudify
        X = torch.cuda.FloatTensor(score)
        y = torch.cuda.FloatTensor(y)

        return X, y

    def __len__(self):
        return self.n_data



def Process_Data(instr, exp_dir, batch_size=16):
    print("loading training data")
    train_dataset = DatasetPreprocessRealTime('data/train_data.hdf5', instr, exp_dir, 'train')
    print("loading test data")
    test_dataset = DatasetPreprocessRealTime('data/train_data.hdf5', instr, exp_dir, 'test')
    test_dataset.write_test_data()

    kwargs = {}
    train_loader = utils.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = utils.DataLoader(test_dataset, batch_size=batch_size, **kwargs)
    return train_loader, test_loader

# def Process_Data(instr, exp_dir):
#     dataset = h5py.File('data/train_data.hdf5','r')
#     score = dataset['{}_pianoroll'.format(instr)][:]
#     spec = dataset['{}_spec'.format(instr)][:]
#     onoff = dataset['{}_onoff'.format(instr)][:]
    # score = np.concatenate((score, onoff),axis = -1)
    # score = np.transpose(score,(0,2,1))

#     X_train, X_test, Y_train, Y_test = train_test_split(score, spec, test_size=0.2, random_state=42) 
    
    # test_data_dir = os.path.join(exp_dir,'test_data')
    # os.makedirs(test_data_dir)
    
    # np.save(os.path.join(test_data_dir, "test_X.npy"), X_test)
    # np.save(os.path.join(test_data_dir, "test_Y.npy"), Y_test)    
    
#     train_dataset = utils.TensorDataset(torch.Tensor(X_train, device=cuda), torch.Tensor(Y_train, device=cuda))
#     train_loader = utils.DataLoader(train_dataset, batch_size=16, shuffle=True)
#     test_dataset = utils.TensorDataset(torch.Tensor(X_test, device=cuda), torch.Tensor(Y_test,device=cuda))
#     test_loader = utils.DataLoader(test_dataset, batch_size=16, shuffle=True) 
    
#     return train_loader, test_loader

def train(model, epoch, train_loader, optimizer,iter_train_loss):
    model.train()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):        
        optimizer.zero_grad()
        split = torch.split(data, 128, dim=1)
        y_pred = model(split[0].cuda(),split[1].cuda())
        loss_function = nn.MSELoss()
        loss = loss_function(y_pred, target.cuda())
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
            y_pred = model(split[0].cuda(),split[1].cuda())
            loss_function = nn.MSELoss() 
            loss = loss_function(y_pred,target.cuda())    
            iter_test_loss.append(loss.item())
            test_loss += loss    
        test_loss/= len(test_loader.dataset)
        scheduler.step(test_loss)
        print ('====> Test set loss: {:.4f}'.format(test_loss))
        return test_loss


def main():    
    hp = hyperparams()

    try:
        exp_root = os.path.join(os.path.abspath('./'),'experiments')
        os.makedirs(exp_root)
    except FileExistsError:
        pass
    
    exp_dir = os.path.join(exp_root, hp.exp_name)
    os.makedirs(exp_dir)

    model = PerformanceNet()
    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    model.zero_grad()
    optimizer.zero_grad()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    train_loader, test_loader = Process_Data(hp.instrument, exp_dir)
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
    main()
