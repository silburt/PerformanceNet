import numpy as np
import librosa.output
import librosa
from intervaltree import Interval,IntervalTree
from scipy import fft 
import pickle
import h5py
import sys
import argparse
import os


class hyperparams(object):
    def __init__(self):
        self.sr = 44100 # Sampling rate.
        self.n_fft = 2048 # fft points (samples)
        self.stride = 256 # 256 samples hop between windows    
        self.wps = 44100 // 256 # ~86 windows/second (for flute?)
        self.instrument = { 
                            'cello': [2217, 2218, 2219, 2220 ,2221, 2222, 2293, 2294],# 2295, 2296, 2297, 2298],
                            'violin': [2191, 2244, 2288, 2289, 2659],
                            'flute':[2202, 2203, 2204]
                            }
        print("warning!! using only a subset of music")
        
        # A.S. each song is chopped into windows, and I *think* hop is the window length?
        # Q: Why do the different instruments have different hop lengths??
        self.hop_inst = {'cello': self.wps, 'violin': int(self.wps * 0.5), 'flute': int(self.wps*0.25)}
                    

hp = hyperparams()


# def get_data(data_dir, inst):
#     '''
    
#     Extract the desired solo data from the dataset.
    
#     Default: 
#         Process cello, violin, flute 
    
#     '''
    
#     dataset = np.load(open(os.path.join(data_dir, 'musicnet.npz'),'rb'), encoding = 'latin1', allow_pickle=True)
#     train_data = h5py.File(os.path.join(data_dir, f'train_data_{inst}.hdf5'), 'w')


#     #for inst in hp.instrument:
#     print ('------ Processing ' + inst + ' ------')
#     score = []
#     audio = []
#     for song in hp.instrument[inst]: 
#         a,b = dataset[str(song)]
#         audio.append(a)
#         score.append(b)

#     spec_list, score_list, onoff_list = process_data(audio,score,inst)
#     train_data.create_dataset(inst + "_spec", data=spec_list)
#     train_data.create_dataset(inst + "_pianoroll", data=score_list)
#     train_data.create_dataset(inst + "_onoff", data=onoff_list)  

def get_data(data_dir, inst):
    '''
    
    Extract the desired solo data from the dataset.
    
    Default: 
        Process cello, violin, flute 
    
    '''
    
    dataset = np.load(open(os.path.join(data_dir, 'musicnet.npz'),'rb'), encoding = 'latin1', allow_pickle=True)
    #train_data = h5py.File(os.path.join(data_dir, f'train_data_{inst}.hdf5'), 'w')
    
    with h5py.File(os.path.join(data_dir, f'train_data_{inst}.hdf5'), 'a') as train_data:
        # get proper dataset chunk sizes
        train_data.create_dataset(inst + "_spec", shape=(0, 1025, 860), dtype='float32', maxshape=(None, 1025, 860)) 
        train_data.create_dataset(inst + "_pianoroll", shape=(0, 860, 128), dtype='float64', maxshape=(None, 860, 128)) 
        train_data.create_dataset(inst + "_onoff", shape=(0, 860, 128), dtype='float64', maxshape=(None, 860, 128)) 

        print ('------ Processing ' + inst + ' ------')
        for index, song in enumerate(hp.instrument[inst]): 
            audio, score = dataset[str(song)]

            spec_list, score_list, onoff_list = process_data([audio], [score], inst)

            train_data[inst + "_spec"].resize((train_data[inst + "_spec"].shape[0] + spec_list.shape[0],) + spec_list.shape[1:])
            train_data[inst + "_spec"][-train_data[inst + "_spec"].shape[0]:] = spec_list

            train_data[inst + "_pianoroll"].resize((train_data[inst + "_pianoroll"].shape[0] + score_list.shape[0],) + score_list.shape[1:])
            train_data[inst + "_pianoroll"][-train_data[inst + "_pianoroll"].shape[0]:] = score_list

            train_data[inst + "_onoff"].resize((train_data[inst + "_onoff"].shape[0] + onoff_list.shape[0],) + onoff_list.shape[1:])
            train_data[inst + "_onoff"][-train_data[inst + "_onoff"].shape[0]:] = onoff_list

            #train_data.create_dataset(inst + "_spec", data=spec_list)
            #train_data.create_dataset(inst + "_pianoroll", data=score_list)
            #train_data.create_dataset(inst + "_onoff", data=onoff_list)  


def process_data(X, Y, inst):
    '''
    Data Pre-processing
        
    Score: 
        Generate pianoroll from interval tree data structure
    
    Audio: 
        Convert waveform into power spectrogram

    '''
    def process_spectrum(X, step, hop):
        audio = X[i][(step * hop * hp.stride): (step * hop * hp.stride) + ((hp.wps*5 - 1)* hp.stride)] 
        spec = librosa.stft(audio, n_fft= hp.n_fft, hop_length = hp.stride)
        magnitude = np.log1p(np.abs(spec)**2)
        return magnitude

    def process_score(Y, step, hop):
        # A.S. 128 dims for the number of notes
        # wps*5 = windows_per_second * 5 is the length of sample time as stated in the paper (arbitrary), length "T" in the paper
        score = np.zeros((hp.wps*5, 128))
        onset = np.zeros(score.shape)    
        offset = np.zeros(score.shape) 

        for window in range(score.shape[0]):
            
            #For score, set all notes to 1 if they are played at this window timestep
            labels = Y[i][(step * hop + window) * hp.stride]
            for label in labels:
                score[window,label.data[1]] = 1
        
            #For onset/offset, set onset to 1 and offset to -1 
            if window != 0:
                onset[window][np.setdiff1d(score[window].nonzero(), score[window-1].nonzero())] = 1
                offset[window][np.setdiff1d(score[window-1].nonzero(), score[window].nonzero())] = -1                    
            else:
                onset[window][score[window].nonzero()] = 1
        
        
        onset += offset 
        return score, onset
    

    spec_list=[]
    score_list=[]
    onoff_list=[]
    num_songs = len(X)
    hop = hp.hop_inst[inst]
    for i in range(num_songs):
        song_length = len(X[i])
        num_spec = (song_length) // (hop * hp.stride)   # A.S. number of spectrograms per song
        print ('{} song {} has {} windows'.format(inst, i, num_spec))

        for step in range(num_spec - 30):   # A.S. why -30?
            if step % 50 == 0:
                print ('{} steps of {} song {} has been done'.format(step,inst,i))        
            spec_list.append(process_spectrum(X,step,hop))
            score, onoff = process_score(Y,step,hop)
            score_list.append(score)
            onoff_list.append(onoff)

    return np.array(spec_list), np.array(score_list), np.array(onoff_list)


def main(args):
    get_data(args.data_dir, args.instrument)
   

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-data-dir", type=str, help="directory where musicnet.npz is")
    parser.add_argument("-instrument", type=str, help="type of instrument to process data for")
    args = parser.parse_args()
    
    main(args)
    
