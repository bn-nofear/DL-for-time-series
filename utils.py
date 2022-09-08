import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import pandas as pd

def get_random_occlusion_mask(dataset, n_intervals, occlusion_prob):
    len_dataset, _, n_features = dataset.shape

    interval_size = int(np.ceil(len_dataset/n_intervals))
    mask = np.ones(dataset.shape)
    for i in range(n_intervals):
        u = np.random.rand(n_features)
        mask_interval = (u>occlusion_prob)*1
        mask[i*interval_size:(i+1)*interval_size, :, :] = mask[i*interval_size:(i+1)*interval_size, :, :]*mask_interval
    # 在每个i处会生成一组随机数，由于prob设为0，所以所有的都不会变为0，这样mask向量全为1

    # Add one random interval for complete missing features 
    feature_sum = mask.sum(axis=0)
    missing_features = np.where(feature_sum==0)[1]
    for feature in missing_features:
        i = np.random.randint(0, n_intervals)
        mask[i*interval_size:(i+1)*interval_size, :, feature] = 1

    return mask

def plot_z(z, filename):
    z_features = z.shape[1]
    fig, ax = plt.subplots(z_features, 1, figsize=(15, z_features))
    for i in range(z_features):
        temp = z[:,i,0,0]
        ax[i].plot(temp)
        ax[i].grid()

    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()
    plt.close('all')
    
def plot_reconstruction_ts(x, x_hat, n_features, filename):
    if n_features>1:
        fig, ax = plt.subplots(n_features, 1, figsize=(15, n_features))
        for i in range(n_features):
            ax[i].plot(x[i])
            ax[i].plot(x_hat[i])
            ax[i].grid()
    else:
        fig = plt.figure(figsize=(15,6))
        plt.plot(x[0])
        plt.plot(x_hat[0])
        plt.grid()
        
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()
    plt.close('all')

def get_auto_corr(timeSeries,k):
    l = len(timeSeries)
    timeSeries1 = timeSeries[0:l-k]
    timeSeries2 = timeSeries[k:]
    timeSeries_mean = timeSeries.mean()
    timeSeries_var = np.array([i**2 for i in timeSeries-timeSeries_mean]).sum()
    auto_corr = 0
    for i in range(l-k):
        temp = (timeSeries1[i]-timeSeries_mean)*(timeSeries2[i]-timeSeries_mean)/timeSeries_var
        auto_corr = auto_corr + temp  
    return auto_corr

# 输入每个tick的x和x_hat, 输出日内均值的x和x_hat
def return_average(x, x_hat, n_features):
    X = []
    X_hat = []
    for i in range(n_features):
        temp = np.array(x[i])
        temp = temp.reshape((-1, 240))
        temp = temp.mean(axis=0)
        X.append(temp)
        temp = np.array(x_hat[i])
        temp = temp.reshape((-1, 240))
        temp = temp.mean(axis=0)
        X_hat.append(temp)
    return X, X_hat

def de_unfold(x_windows, mask_windows, window_step):
    """
    x_windows of shape (n_windows, n_features, 1, window_size)
    mask_windows of shape (n_windows, n_features, 1, window_size)
    """
    n_windows, n_features, _, window_size = x_windows.shape

    assert (window_step == 1) or (window_step == window_size), 'Window step should be either 1 or equal to window_size'

    len_series = (n_windows)*window_step + (window_size-window_step)
    # print(len_series)
    x = np.zeros((len_series, n_features))
    # print(x.shape)
    mask = np.zeros((len_series, n_features))
    # print(mask.shape)
    n_windows = len(x_windows)
    for i in range(n_windows):
        x_window = x_windows[i,:,0,:]
        x_window = np.swapaxes(x_window,0,1)
        x[i*window_step:(i*window_step+window_size),:] += x_window

        mask_window = mask_windows[i,:,0,:]
        mask_window = np.swapaxes(mask_window,0,1)
        mask[i*window_step:(i*window_step+window_size),:] += mask_window

    division_safe_mask = mask.copy()
    division_safe_mask[division_safe_mask==0]=1
    x = x/division_safe_mask
    mask = 1*(mask>0)
    return x, mask

def analyze(x, x_hat, n_features, filename):
    result = pd.DataFrame(columns=['true_mean','true_var','true_autocorr','hat_mean','hat_var','hat_autocorr','correlation','R2_Square'])
    for i in range(n_features):
        temp1 = np.array(x[i])
        temp2 = np.array(x_hat[i])
        SST = np.array([i**2 for i in (temp1-temp1.mean())]).sum()
        SSE = np.array([i**2 for i in (temp1-temp2)]).sum()
        series = pd.Series({'true_mean':temp1.mean(),'true_var':temp1.var(),'true_autocorr':get_auto_corr(temp1,240),'hat_mean':temp2.mean(),'hat_var':temp2.var(),'hat_autocorr':get_auto_corr(temp2,240),'correlation':np.corrcoef(temp1,temp2)[0][1],'R2_Square':1-SSE/SST},name=('Stock_'+str(i+1)))
        result = result.append(series)
    result.to_csv(filename)