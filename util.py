import time,os
import numpy as np
import torch
import random
from torch.optim import lr_scheduler
import math


def evaluate(r_ref, r_ans,fs =200, thr=30,print_msg = True):
    
    all_TP = 0
    all_FN = 0
    all_FP = 0
    tol = int(thr*fs/1000)
    errors = []
    for i in range(len(r_ref)):
        FN = 0
        FP = 0
        TP = 0
        detect_loc = 0
        for j in range(len(r_ref[i])):
            loc = np.where(np.abs(r_ans[i] - r_ref[i][j]) <= tol)[0]
            detect_loc += len(loc)

            if len(loc) >= 1:
                
                TP += 1
                FP = FP + len(loc) - 1
                
                diff = r_ref[i][j] - r_ans[i][loc[0]]
                errors.append(diff/fs)
                
            elif len(loc) == 0:
                FN += 1
        FP = FP+(len(r_ans[i])-detect_loc)
        
        all_FP += FP
        all_FN += FN
        all_TP += TP
    if all_TP == 0:
        Recall = 0
        Precision = 0
        F1_score = 0
        Sen = 0
        PPV = 0
        
    else:
        Sen = all_TP / (all_FN + all_TP)
        PPV = 0
        Recall = all_TP / (all_FN + all_TP)
        Precision = all_TP / (all_FP + all_TP )
        F1_score = 2 * Recall * Precision / (Recall + Precision)
    if all_FP == 0:
        error_rate = 0
    else:
        error_rate =  all_FP / (all_FP + all_TP)
    if print_msg:
        print("TP's:{} FN's:{} FP's:{}".format(all_TP,all_FN,all_FP))
        print('Recall:{},Precision:{},F1-score:{}'.format(Recall,Precision, F1_score))
    
    return Recall,Precision, F1_score


def get_label(peaks,length,fs = 1000, tol = 17):
    
    half = tol//2
    labels = np.zeros(length)
    for peak in peaks:
        if peak - half >= 0 and peak + half +1 < length:
            for i in range(peak-half, peak + half + 2):
                labels[i] = 1
    labels = labels.reshape((1, length))
    return labels

def get_peak_from_label(prob,fs = 200):
    labels = prob.copy()



    tol = 17    
    h_tol = 8
    labels = labels.flatten()
    length = len(labels)
    a = np.zeros((length))
    for i in range(length):
        for j in range(i-h_tol, i+h_tol+1):
            if j<0 or j >= length:
                continue
            if labels[j] == 1:
                a[i] += 1

    W = 70
    W = int(W)

    peaks = []
    i= 0
    while i < length:
        max_a = -1
        for j in range(i-W, i+W):
            if j<0 or j >= length:
                continue
            if a[j] > max_a:
                max_a = a[j]
        
        if a[i] == max_a and max_a>h_tol:
            dis = 1
            k = i
            while k < length:
                if a[k] == max_a:
                    dis+=1
                    k+=1
                else:
                    break
            loc = i + dis//2
            peaks.append(loc)
            i =loc + tol
        i+=1

    
    return peaks