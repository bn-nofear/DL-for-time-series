import os
import pickle
import numpy as np
import pandas as pd
import torch
from DGHL import DGHL
from utils import de_unfold, plot_reconstruction_ts, get_random_occlusion_mask, compute_R2 ,plot_z, return_average
import sys

data_file_path = r'E:\BaiduNetdiskDownload\stock_data'
execute_path = r'E:\大学\大二下\OurCode'
path_list = os.listdir(data_file_path)
path_list.sort(key=lambda x: int(x[3:-4]))
NUM,DAYS = sys.argv[1:]

def main(num,days):
    os.makedirs(os.path.join(execute_path , str(num) + 'stocks_' + str(days) + 'days'),exist_ok=True)
    new_execute_path = os.path.join(execute_path , str(num) + 'stocks_' + str(days) + 'days')
    df = pd.DataFrame()
    for path in path_list[-(days+10):]:
        df = pd.concat([df,pd.read_csv(os.path.join(data_file_path,path))],axis=0)
    df = df.iloc[:,1:].dropna(axis=0,how='all').dropna(axis=1)
    if len(df.columns) < num or len(df) < 240*days:
        print('Not enough stocks meet the standard!')
    else:
        aim = df.iloc[-days*240:-int(4/5*days)*240,:num]
        aim2 = df.iloc[-int(4/5*days)*240:,:num]
        aim.to_csv(new_execute_path + '/train_data.txt',header=0,index=0)
        aim2.to_csv(new_execute_path + '/test_data.txt',header=0,index=0)
        
        
        for entity in range(1):
        # ------------------------------------------------- Parameters -------------------------------------------------
            mc = {}
            mc['window_size'] = 10
            mc['window_step'] = 240
            mc['n_features'] = num
            mc['hidden_multiplier'] = 32
            mc['max_filters'] = 256
            mc['kernel_multiplier'] = 1
            mc['z_size'] = 20
            mc['z_size_m'] = 10
            mc['z_size_up'] = 5
            mc['window_hierarchy'] = 24
            mc['middle_window_hierarchy'] = 2
            mc['z_iters'] = 25
            mc['z_sigma'] = 0.25
            mc['z_step_size'] = 0.1
            mc['z_with_noise'] = False
            mc['z_persistent'] = True
            mc['z_iters_inference'] = 100
            mc['batch_size'] = 4
            mc['learning_rate'] = 1e-3
            mc['noise_std'] = 0.001
            mc['n_iterations'] = 100
            mc['normalize_windows'] = False
            mc['random_seed'] = 1
            mc['device'] = None
            mc['occlusion_intervals'] = 1
            mc['occlusion_prob'] = 0.2
            mc['downsampling_size'] = 20
            print(pd.Series(mc))

            name = 'data.txt'

        # ------------------------------------------------- Reading data -------------------------------------------------
            train_data = np.loadtxt(new_execute_path + '/train_data.txt', delimiter=',')
            train_data = train_data[:, None, :]

            test_data = np.loadtxt(new_execute_path + '/test_data.txt',delimiter=',') 
            test_data = test_data[:, None, :]

            dataset = np.vstack([train_data, test_data])

            print('Full Train shape: ', train_data.shape)
            print('Full Test shape: ', test_data.shape)
            print('Full Dataset shape: ', dataset.shape)

        # ------------------------------------------------- Downsampling -------------------------------------------------
            # downsampling_size = mc['downsampling_size']
            # len_test = len(test_data)
            # right_padding = downsampling_size - dataset.shape[0]%downsampling_size
            # dataset = np.pad(dataset, ((right_padding, 0), (0,0), (0,0) ))

            # right_padding = downsampling_size - len_test%downsampling_size
            
            # dataset = dataset.reshape(dataset.shape[0]//downsampling_size, -1, 1, dataset.shape[2]).max(axis=1)
            # len_test_downsampled = int(np.ceil(len_test/downsampling_size))
            
            # print('Downsampled Dataset shape: ', dataset.shape)

            # train_data = dataset[:-len_test_downsampled]
            # test_data = dataset[-len_test_downsampled:]

        # ------------------------------------------------- Training Occlusion Mask -------------------------------------------------
            train_mask = get_random_occlusion_mask(dataset=train_data, n_intervals=mc['occlusion_intervals'], occlusion_prob=mc['occlusion_prob'])

        # --------------------------------------- Random seed --------------------------------------
            np.random.seed(mc['random_seed'])

        # --------------------------------------- Parse paramaters --------------------------------------
            window_size = mc['window_size']
            window_hierarchy = mc['window_hierarchy']
            window_step = mc['window_step']
            n_features = mc['n_features']

            total_window_size = window_size*window_hierarchy
        # --------------------------------------- Data Processing ---------------------------------------
            # Complete first window for test, padding from training data
            padding = total_window_size - (len(test_data) - total_window_size*(len(test_data)//total_window_size))
            test_data = np.vstack([train_data[-padding:], test_data])
            test_mask = np.ones(test_data.shape) #无mask
            test_mask = np.vstack([train_mask[-padding:], test_mask])

            # Create rolling windows
            train_data = torch.Tensor(train_data).float()
            train_data = train_data.permute(0,2,1)
            train_data = train_data.unfold(dimension=0, size=total_window_size, step=window_step)

            test_data = torch.Tensor(test_data).float()
            test_data = test_data.permute(0,2,1)
            test_data = test_data.unfold(dimension=0, size=total_window_size, step=window_step)

            train_mask = torch.Tensor(train_mask).float()
            train_mask = train_mask.permute(0,2,1)
            train_mask = train_mask.unfold(dimension=0, size=total_window_size, step=window_step)

            test_mask = torch.Tensor(test_mask).float()
            test_mask = test_mask.permute(0,2,1)
            test_mask = test_mask.unfold(dimension=0, size=total_window_size, step=window_step)

            print('Windowed Train shape: ', train_mask.shape)
            print('Windowed Test shape: ', test_mask.shape)

        # -------------------------------------------- Instantiate and train Model --------------------------------------------
            print('Training model...')
            model = DGHL(window_size=window_size, window_step=mc['window_step'], window_hierarchy=window_hierarchy,
                        hidden_multiplier=mc['hidden_multiplier'], max_filters=mc['max_filters'],
                        kernel_multiplier=mc['kernel_multiplier'], n_channels=n_features,
                        z_size=mc['z_size'], z_size_m=mc['z_size_m'], z_size_up=mc['z_size_up'], z_iters=mc['z_iters'],
                        z_sigma=mc['z_sigma'], z_step_size=mc['z_step_size'],
                        z_with_noise=mc['z_with_noise'], z_persistent=mc['z_persistent'],
                        batch_size=mc['batch_size'], learning_rate=mc['learning_rate'],
                        noise_std=mc['noise_std'],middle_window_hierarchy=mc['middle_window_hierarchy'],
                        normalize_windows=mc['normalize_windows'],
                        random_seed=mc['random_seed'], device=mc['device'])

            model.fit(X=train_data, mask=train_mask, n_iterations=mc['n_iterations'], plot_mse=True)

        # -------------------------------------------- Inference on each entity --------------------------------------------
            # rootdir_entity = 'Results'
            # os.makedirs(name=rootdir_entity, exist_ok=True)
            # Plots of reconstruction in train
            print('Reconstructing train...')
            x_train_true, x_train_hat, z, mask_windows = model.predict(X=train_data, mask=train_mask,
                                                                        z_iters=mc['z_iters_inference'])
            plot_z(z, filename= new_execute_path + '/latent variables.png')
            
            # 得(n_window, n_features, 1, window_size*window_hierarchy size)

            # 把window切片展开为原来的序列
            x_train_true, _ = de_unfold(x_windows=x_train_true, mask_windows=mask_windows, window_step=window_step)
            x_train_hat, _ = de_unfold(x_windows=x_train_hat, mask_windows=mask_windows, window_step=window_step)
            # (n_time ,n_featrues)

            x_train_true = np.swapaxes(x_train_true,0,1)
            x_train_hat = np.swapaxes(x_train_hat,0,1)
            # (n_features, n_time)
            
            filename = new_execute_path + '/reconstruction_train.png'
            plot_reconstruction_ts(x=x_train_true, x_hat=x_train_hat, n_features=n_features, filename=filename)
            print("R2 of train:",compute_R2(x=x_train_true, x_hat=x_train_hat ,n_features=n_features))

        # --------------------------------------- Inference on test and anomaly scores ---------------------------------------
            print('Computing scores on test...')
            x_test_true, x_test_hat, z, mask_windows = model.predict(X=test_data, mask=test_mask,
                                                                        z_iters=mc['z_iters_inference'])
            
            # 得(n_window, n_features, 1, window_size*window_hierarchy size)
                
            # 把window切片展开为原来的序列
            x_test_true, _ = de_unfold(x_windows=x_test_true, mask_windows=mask_windows, window_step=window_step)
            x_test_hat, _ = de_unfold(x_windows=x_test_hat, mask_windows=mask_windows, window_step=window_step)
            # (n_time ,n_featrues)

            x_test_true = np.swapaxes(x_test_true,0,1)
            x_test_hat = np.swapaxes(x_test_hat,0,1)
            # (n_features, n_time)

            filename = new_execute_path + '/reconstruction_test.png'
            plot_reconstruction_ts(x=x_test_true, x_hat=x_test_hat, n_features=n_features, filename=filename)
            pd.DataFrame(compute_R2(x=x_test_true, x_hat=x_test_hat ,n_features=n_features)).to_csv(new_execute_path + '/R2.txt')
            filename = new_execute_path + '/reconstruction_test_average.png'
            x_test_true, x_test_hat = return_average(x_test_true, x_test_hat, n_features)
            plot_reconstruction_ts(x=x_test_true, x_hat=x_test_hat, n_features=n_features, filename=filename)

if __name__ == "__main__":
    main(num=int(NUM),days=int(DAYS))