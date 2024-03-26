import pywt, os, copy
import torch
import numpy as np
import pandas as pd
from config import config
from torch.utils.data import Dataset
from sklearn.preprocessing import scale
from sklearn import preprocessing
import data_util as du
import scipy
from scipy.signal import butter, filtfilt
import util

def iirnotch(data, fs):
    w3, w4 = scipy.signal.iirnotch(50, 50, fs=fs)  # 50hz陷波滤波器
    w5, w6 = scipy.signal.iirnotch(60, 50, fs=fs)  # 60hz陷波滤波器
    
    data = scipy.signal.filtfilt(w3, w4, data, axis = 1)
    data = scipy.signal.filtfilt(w5, w6, data, axis = 1)
    return data



def get_namelist(train=True, test_idx = 0):
    namelist = du.get_namelist()
  
    if train:
        new_list = []
        for i in range(len(namelist)):
            if test_idx == i:
                continue
            new_list.append(namelist[i])
    else:
        new_list = [namelist[test_idx]]
        
    
    return new_list


def windowingSig(sig1, labels, windowSize=1000):
  
    sig1 = du.pad_audio(sig1,windowSize)
    signalsWindow1 = du.enframe(sig1,windowSize)
    labels = du.pad_audio(labels,windowSize)
    labelsWindow = du.enframe(labels,windowSize)

    return signalsWindow1, labelsWindow




class Dataset(Dataset):
    def __init__(self, train=True, seg_len = 600,fs = 200,test_idx = 0):
        
        super(Dataset, self).__init__()
        
        
        self.origin_fs = 1000
        self.fs = fs
       
        self.namelist = get_namelist(train, test_idx)
       
        self.test_idx = test_idx

      
        
        ecgWindows,labelsWindows, self.fqrs_rpeaks = self.prepareData(train=train, seg_len = seg_len)
        self.X_train, self.Y_train = np.array(ecgWindows),np.array(labelsWindows) #np.array(fecgWindows)
        
    
    def get_fqrs(self):
        return self.fqrs_rpeaks
        
    def readData(self, name):

        ecg,peaks = du.get_data(name)
       
        ecg = self.preprocess(ecg, self.origin_fs)
        scale_fecg_fs = self.origin_fs/self.fs
        peaks = np.asarray(np.floor_divide(peaks,scale_fecg_fs),'int64')

        return ecg,peaks

    def preprocess(self, signal, origin_fs):
        signal = self.butter_bandpass_filter(signal, 7.5, 75, origin_fs)

        signal = iirnotch(signal, origin_fs)
        signal =scale(signal, axis=1)
        scale_fs = origin_fs/self.fs

        signal = scipy.signal.resample(signal, int(signal.shape[1] / scale_fs), axis=1)

        return signal
    
    
    def prepareData(self, train=True, seg_len = 600):
        ecgAll, labels,peakAll= None, None,None
        overlap = True
        cnt = 0
        pad = False
 
        for name in self.namelist:
          
         
            ecg, peaks = self.readData(name)
            ecg_len = ecg.shape[1]

            label = util.get_label(peaks, ecg_len, fs = self.fs,tol = 17)


            if ecgAll is None:
                ecgAll = ecg

                labels = label
                peakAll = peaks
                
            else:
                ecgAll = np.append(ecgAll, ecg, axis=1)

                labels = np.append(labels, label, axis=1)
                peakAll = np.append(peakAll, peaks+cnt*ecg_len)
            
            cnt += 1

        ecgWindows, labelWindows = windowingSig(ecgAll, labels, windowSize=seg_len)
        ecgWindows = np.asarray(ecgWindows)

        labelWindows = np.asarray(labelWindows)

        return ecgWindows,labelWindows, peakAll
             



    def __butter_bandpass(self, lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def butter_bandpass_filter(self, data, lowcut, highcut, fs, order=3, axis=1):
        b, a = self.__butter_bandpass(lowcut, highcut, fs, order=order)
        y = filtfilt(b, a, data, axis=axis)
        return y
    
    
    
    

    def __getitem__(self, index):        
        dataset_x = self.X_train[index,:,:]
        dataset_y = self.Y_train[index,:,:]
        return dataset_x, dataset_y


    def __len__(self):
        return self.X_train.shape[0]
    
