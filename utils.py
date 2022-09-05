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

    # Add one random interval for complete missing features 
    feature_sum = mask.sum(axis=0)
    missing_features = np.where(feature_sum==0)[1]
    for feature in missing_features:
        i = np.random.randint(0, n_intervals)
        mask[i*interval_size:(i+1)*interval_size, :, feature] = 1

    return mask

def plot_z(z, filename):
    df = pd.DataFrame()
    z_features = z.shape[1]
    fig, ax = plt.subplots(z_features, 1, figsize=(15, z_features))
    for i in range(z_features):
        temp = z[:,i,0,0]
        ax[i].plot(temp)
        ax[i].grid()
        df[i] = z[:,i,0,0]

    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()
    plt.close('all')
    filename2 = filename[:-4]+'.csv'
    df.to_csv(filename2)
    
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

    x = np.zeros((len_series, n_features))
    mask = np.zeros((len_series, n_features))

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

def compute_R2(x, x_hat, n_features):
    # 对每个feature分别求R2, 返回一个列表
    mse = nn.MSELoss(reduction='sum')
    R2 = []
    x, x_hat = torch.Tensor(x), torch.Tensor(x_hat)
    for i in range(n_features):
        print(x_hat[i].shape)
        R = 1 - mse(x[i], x_hat[i])/(torch.var(x[i])*(len(x[i])-1))
        R2.append(R)
    return R2