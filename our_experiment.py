from operator import truediv
import os
import pickle
import numpy as np
import pandas as pd
import torch
from DGHL import DGHL
from utils import de_unfold, plot_reconstruction_ts, get_random_occlusion_mask, analyze, plot_z, return_average
import sys

data_file_path = r'D:\stock_data'
execute_path = r'C:\Users\30861\Desktop\OurCode'
path_list = os.listdir(data_file_path)
path_list.sort(key=lambda x: int(x[3:-4]))
NUM,DAYS = sys.argv[1:]

def our_experiment(num,days):
    
    # ------------------------------------------------- Data preprocessing -------------------------------------------------
        new_execute_path = os.path.join(execute_path , str(num) + 'stocks_' + str(days) + 'days')
        os.makedirs(new_execute_path,exist_ok=True)
        df = pd.DataFrame()
        for path in path_list[-(days+10):]:
            df = pd.concat([df,pd.read_csv(os.path.join(data_file_path,path))],axis=0)
        df = df.iloc[:,1:].dropna(axis=0,how='all').dropna(axis=1)

        assert (len(df.columns) >= num) and (len(df) >= 240*days)

        Train = df.iloc[-days*240:-int(1/5*days)*240,:num]
        Test = df.iloc[-int(1/5*days)*240:,:num]

        train_data = np.array(Train)
        train_data = train_data[:, None, :]

        test_data = np.array(Test)
        test_data = test_data[:, None, :]

        dataset = np.vstack([train_data, test_data])

        print('Full Train shape: ', train_data.shape) 
        print('Full Test shape: ', test_data.shape)
        print('Full Dataset shape: ', dataset.shape)

    # ------------------------------------------------- Parameters -------------------------------------------------
        mc = {}
        mc['window_size'] = 5 # 一个subwindow的tick数
        mc['window_step'] = 240 # 一个window的tick数
        mc['n_features'] = num
        mc['hidden_multiplier'] = 32
        mc['max_filters'] = 256
        mc['kernel_multiplier'] = 1
        mc['z_size'] = [5, 10, 20, 20] # 从上到下的隐变量维数
        mc['window_hierarchy'] = [1, 4, 24, 48] # 从上到下的隐变量每层个数 一天、每小时、十分钟、五分钟
        mc['z_iters'] = 25
        mc['z_sigma'] = 0.25
        mc['z_step_size'] = 0.1
        mc['z_with_noise'] = True
        mc['z_persistent'] = True
        mc['z_iters_inference'] = 100
        mc['batch_size'] = 4
        mc['learning_rate'] = 1e-3
        mc['noise_std'] = 0.001
        mc['n_iterations'] = 500
        mc['normalize_windows'] = False
        mc['random_seed'] = 1
        mc['device'] = None
        mc['occlusion_intervals'] = 1
        mc['occlusion_prob'] = 0.3
        mc['downsampling_size'] = 20
        print(pd.Series(mc))

    # ------------------------------------------------- Downsampling -------------------------------------------------
        # downsampling_size = mc['downsampling_size']
        # len_test = len(test_data)
        # right_padding = downsampling_size - dataset.shape[0]%downsampling_size
        # dataset = np.pad(dataset, ((right_padding, 0), (0,0), (0,0) ))   # 这一步在向前填充0 shape变为(1220,1,2)

        # right_padding = downsampling_size - len_test%downsampling_size
        
        # dataset = dataset.reshape(dataset.shape[0]//downsampling_size, -1, 1, dataset.shape[2]).max(axis=1) # 在max之前shape为(61,20,1,2)后为(61,1,2)
        # len_test_downsampled = int(np.ceil(len_test/downsampling_size))
        
        # print('Downsampled Dataset shape: ', dataset.shape)

        # train_data = dataset[:-len_test_downsampled] # train的shape为(49,1,2)
        # test_data = dataset[-len_test_downsampled:] # test的shape为(12,1,2)
    # ------------------------------------------------- Training Occlusion Mask -------------------------------------------------
        train_mask = get_random_occlusion_mask(dataset=train_data, n_intervals=mc['occlusion_intervals'], occlusion_prob=mc['occlusion_prob'])
        # mask的shape为(61,1,2)
    # --------------------------------------- Random seed --------------------------------------
        np.random.seed(mc['random_seed'])

    # --------------------------------------- Parse paramaters --------------------------------------
        window_size = mc['window_size']
        window_hierarchy = mc['window_hierarchy']
        window_step = mc['window_step']
        n_features = mc['n_features']
        total_window_size = window_step

    # --------------------------------------- Data Processing ---------------------------------------
        # Complete first window for test, padding from training data
        padding = total_window_size - (len(test_data) - total_window_size*(len(test_data)//total_window_size))
        test_data = np.vstack([train_data[-padding:], test_data])
        # test_mask = np.ones(test_data.shape) #无mask
        # test_mask = np.vstack([train_mask[-padding:], test_mask])
        u = test_data.shape
        test_mask_1 = np.ones((u[0]-int(u[0]/3), u[1], u[2]))
        test_mask_2 = np.zeros((int(u[0]/3), u[1], u[2]))
        test_mask = np.concatenate([test_mask_1, test_mask_2])
        test_mask = np.vstack([train_mask[-padding:], test_mask])

        # Create rolling windows
        train_data = torch.Tensor(train_data).float()
        train_data = train_data.permute(0,2,1) # 维度转换为(49/960,2,1)
        train_data = train_data.unfold(dimension=0, size=total_window_size, step=window_step) # 维度为(4,2,1,240)
        
        # 其他也对应展开
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
                    z_size=mc['z_size'], z_iters=mc['z_iters'],
                    z_sigma=mc['z_sigma'], z_step_size=mc['z_step_size'],
                    z_with_noise=mc['z_with_noise'], z_persistent=mc['z_persistent'],
                    batch_size=mc['batch_size'], learning_rate=mc['learning_rate'],
                    noise_std=mc['noise_std'],
                    normalize_windows=mc['normalize_windows'],
                    random_seed=mc['random_seed'], device=mc['device'])

        model.fit(X=train_data, mask=train_mask, n_iterations=mc['n_iterations'], plot_mse=True, path=new_execute_path)
        # model.fit(X=train_data, mask=train_mask, n_iterations=mc['n_iterations'], plot_mse=True)

    # -------------------------------------------- Inference on each entity --------------------------------------------
        # Plots of reconstruction in train
        print('Reconstructing train...')
        x_train_true, x_train_hat, z_train, mask_windows = model.predict(X=train_data, mask=train_mask,
                                                                    z_iters=mc['z_iters_inference'])
        plot_z(z_train, filename= new_execute_path + '/latent variables_train.png')
        # 得(n_window, n_features, 1, window_size*window_hierarchy size)

        # 把window切片展开为原来的序列
        x_train_true, _ = de_unfold(x_windows=x_train_true, mask_windows=mask_windows, window_step=window_step)
        x_train_hat, _ = de_unfold(x_windows=x_train_hat, mask_windows=mask_windows, window_step=window_step)
        # (n_time ,n_featrues)

        x_train_true = np.swapaxes(x_train_true,0,1)
        x_train_hat = np.swapaxes(x_train_hat,0,1)
        print(x_train_true.shape,x_train_hat.shape)
        # (n_features, n_time)
        
        filename = new_execute_path + '/reconstruction_train.png'
        plot_reconstruction_ts(x=x_train_true, x_hat=x_train_hat, n_features=n_features, filename=filename)

        filename = new_execute_path + '/train_result.csv'
        analyze(x=x_train_true, x_hat=x_train_hat, n_features=n_features, filename=filename)

    # --------------------------------------- Inference on test and anomaly scores ---------------------------------------
        print('Computing scores on test...')
        x_test_true, x_test_hat, z_test, mask_windows = model.predict(X=test_data, mask=test_mask,
                                                                    z_iters=mc['z_iters_inference'])
        plot_z(z_test, filename= new_execute_path + '/latent variables_test.png')
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

        filename = new_execute_path + '/test_result.csv'
        analyze(x=x_test_true, x_hat=x_test_hat, n_features=n_features, filename=filename)

        filename = new_execute_path + '/reconstruction_test_average.png'
        x_test_true, x_test_hat = return_average(x_test_true, x_test_hat, n_features)
        plot_reconstruction_ts(x=x_test_true, x_hat=x_test_hat, n_features=n_features, filename=filename)

if __name__ == "__main__":
    our_experiment(num=int(NUM),days=int(DAYS))
    # for NUM in [2,5,10,20]:
    #     for DAYS in [10,20,50,100,200]:
    #         main(num=NUM,days=DAYS)