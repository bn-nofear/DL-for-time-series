import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision as tv
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, window_size=32, hidden_multiplier=32, latent_size=100, n_channels=3, max_filters=256, kernel_multiplier=1):
        super(Generator, self).__init__()

        n_layers = int(np.log2(window_size))+1
        layers = []
        filters_list = []
        # First layer
        filters = min(max_filters, hidden_multiplier*(2**(n_layers-2)))
        layers.append(nn.ConvTranspose1d(in_channels=latent_size, out_channels=filters,
                                         kernel_size=4, stride=1, padding=0, bias=False))
        layers.append(nn.BatchNorm1d(filters))
        filters_list.append(filters)
        # Hidden layers
        for i in reversed(range(1, n_layers-1)):
            filters = min(max_filters, hidden_multiplier*(2**(i-1)))
            layers.append(nn.ConvTranspose1d(in_channels=filters_list[-1], out_channels=filters,
                                             kernel_size=4*kernel_multiplier, stride=2, padding=1 + (kernel_multiplier-1)*2, bias=False))
            layers.append(nn.BatchNorm1d(filters))
            layers.append(nn.ReLU())
            filters_list.append(filters)

        # Output layer
        layers.append(nn.ConvTranspose1d(in_channels=filters_list[-1], out_channels=n_channels, kernel_size=3, stride=1, padding=1))
        self.layers = nn.Sequential(*layers)

    def forward(self, x, m=None):
        x = x[:,:,0,:]
        x = self.layers(x)
        x = x[:,:,None,:m.shape[3]]
        
        # Hide mask
        if m is not None:
            x = x * m

        return x

class DGHL(object):
    def __init__(self, window_size, window_step, window_hierarchy,
                 n_channels, hidden_multiplier, max_filters, kernel_multiplier,
                 z_size, z_iters, z_sigma, z_step_size, z_with_noise, z_persistent,
                 batch_size, learning_rate, noise_std, normalize_windows,
                 random_seed, device=None):
        super(DGHL, self).__init__()

        # Generator
        self.window_size = window_size # Not used for now
        self.window_step = window_step # Not used for now
        self.n_channels = n_channels
        self.hidden_multiplier = hidden_multiplier
        self.z_size = z_size
        self.max_filters = max_filters
        self.kernel_multiplier = kernel_multiplier
        self.normalize_windows = normalize_windows

        self.window_hierarchy = window_hierarchy

        # Alternating back-propagation
        self.z_iters = z_iters
        self.z_sigma = z_sigma
        self.z_step_size = z_step_size
        self.z_with_noise = z_with_noise
        self.z_persistent = z_persistent

        # Training
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.noise_std = noise_std

        # Generator
        torch.manual_seed(random_seed)

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.generator = Generator(window_size=self.window_size, hidden_multiplier=self.hidden_multiplier,
                                   latent_size=int(sum(self.z_size)),
                                   n_channels=self.n_channels, max_filters=self.max_filters,
                                   kernel_multiplier=self.kernel_multiplier).to(self.device)

    def get_z(self, z, x, m, n_iters, with_noise):
        mse = nn.MSELoss(reduction='sum')
        temp = [z[j] for j in range(len(self.z_size))]
        for i in range(n_iters):
            temp_repeated = []
            for j in range(len(self.z_size)):
                temp[j] = torch.autograd.Variable(temp[j], requires_grad=True)
                temp_repeated.append(torch.repeat_interleave(temp[j], int(self.window_hierarchy[-1]/self.window_hierarchy[j]), 0))
                if (j==0):
                    z = temp_repeated[0]
                else:
                    z = torch.cat((z, temp_repeated[j]),dim=1).to(self.device)

            x_hat = self.generator(z, m)

            L = 1.0 / (2.0 * self.z_sigma * self.z_sigma) * mse(x_hat, x)
            L.backward()

            for j in range(len(self.z_size)): # 不理解 为什么这么更新
                temp[j] = temp[j] - 0.5 * self.z_step_size * self.z_step_size * (temp[j] + temp[j].grad)

            if with_noise:
                for j in range(len(self.z_size)):
                    eps = torch.randn(len(temp[j]), self.z_size[j], 1, 1).to(temp[j].device)
                    temp[j] = temp[j] + self.z_step_size * eps

        for j in range(len(self.z_size)):
            temp[j] = temp[j].detach()
        z = z.detach()

        return z, temp

    def sample_gaussian(self, n_dim, n_samples):#返回(n_samples, n_dim, 1, 1)
        p_0 = torch.distributions.MultivariateNormal(torch.zeros(n_dim), 0.01*torch.eye(n_dim))
        p_0 = p_0.sample([n_samples]).view([n_samples, -1, 1, 1])

        return p_0

    def get_batch(self, X, mask, batch_size, p_0_chains, z_persistent, shuffle=False):#(batch_size*window_hierarchy, n_features, 1, window_size)
        """
        X tensor of shape (n_windows, n_features, 1, window_size*A_L)
        """
        if shuffle:
            i = torch.LongTensor(batch_size).random_(0, X.shape[0])
        else:
            i = torch.LongTensor(range(batch_size))

        p_d_x = X[i]
        p_d_m = mask[i]

        x_scales = p_d_x[:,:,:,[0]]
        if self.normalize_windows:
            p_d_x = p_d_x - x_scales

        # Wrangling from (batch_size, n_features, 1, window_size*window_hierarchy[-1]) -> (batch_size*window_hierarchy[-1], n_features, 1, window_size)
        p_d_x = p_d_x.unfold(dimension=-1, size=self.window_size, step=self.window_size)
        p_d_x = p_d_x.swapaxes(1,3)
        p_d_x = p_d_x.swapaxes(2,3)
        p_d_x = p_d_x.reshape(batch_size*self.window_hierarchy[-1], self.n_channels, 1, self.window_size)

        p_d_m = p_d_m.unfold(dimension=-1, size=self.window_size, step=self.window_size)
        p_d_m = p_d_m.swapaxes(1,3)
        p_d_m = p_d_m.swapaxes(2,3)
        p_d_m = p_d_m.reshape(batch_size*self.window_hierarchy[-1], self.n_channels, 1, self.window_size)

        # Hide with mask
        p_d_x = p_d_x * p_d_m
        
        p_0_z = []
        if z_persistent:
            for j in range(len(self.z_size)):
                temp = p_0_chains[j][i]
                temp = temp.reshape(batch_size * self.window_hierarchy[j], self.z_size[j], 1, 1)
                p_0_z.append(temp)
        else:
            for j in range(len(self.z_size)):
                temp = self.sample_gaussian(n_dim=self.z_size[j], n_samples=batch_size*self.window_hierarchy[j])
                temp = temp.to(self.device)
                p_0_z.append(temp)

        p_d_x = torch.Tensor(p_d_x).to(self.device)
        p_d_m = torch.Tensor(p_d_m).to(self.device)
        x_scales = x_scales.to(self.device)
        
        return p_d_x, p_0_z, p_d_m, i, x_scales

    def fit(self, X, mask, n_iterations, plot_mse=True, path=''):
        #如果z_persistent 就在进入iteration前生成好高斯分布，iteration中做细微改变；否则每次iteration都重新随机生成
        # p_0_chains list 从上到下的每一层隐变量
        if self.z_persistent:
            self.p_0_chains = []
            for i in range(len(self.z_size)):
                temp = torch.zeros((X.shape[0], self.window_hierarchy[i], self.z_size[i], 1, 1))
                self.p_0_chains.append(temp)
            for i in range(X.shape[0]):
                for j in range(len(self.z_size)):
                    p_0_chains = self.sample_gaussian(n_dim=self.z_size[j], n_samples=self.window_hierarchy[j])
                    p_0_chains = p_0_chains.to(self.device)
                    self.p_0_chains[j][i] = p_0_chains
            
        else:
            self.p_0_chains  = [None for i in range(len(self.z_size))]

        optim = torch.optim.Adam(self.generator.parameters(), lr=self.learning_rate, betas=[.9, .999])
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=int(np.ceil(n_iterations/3)), gamma=0.8)
        #设置Adam优化器和学习率衰减

        mse = nn.MSELoss(reduction='sum')
        # Training loop
        mse_list = []
        start_time = time.time()
        for i in range(n_iterations):
            self.generator.train()
            # Sample windows
            x, z_0, m, chains_i, x_scales = self.get_batch(X=X, mask=mask, batch_size=self.batch_size, 
                                                 p_0_chains=self.p_0_chains,
                                                 z_persistent=self.z_persistent, shuffle=True)
            x = x + self.noise_std*(torch.randn(x.shape).to(self.device))
            #print(x.shape)
            # Sample z with Langevin Dynamics
            z, temp = self.get_z(z=z_0, x=x, m=m, n_iters=self.z_iters, with_noise=self.z_with_noise)
            x_hat = self.generator(z, m)

            # Return to window_size * window_hierarchy size
            x = x.swapaxes(0,2)
            x = x.reshape(1,self.n_channels,-1, self.window_size*self.window_hierarchy[-1])
            x = x.swapaxes(0,2)

            x_hat = x_hat.swapaxes(0,2)
            x_hat = x_hat.reshape(1,self.n_channels,-1, self.window_size*self.window_hierarchy[-1])
            x_hat = x_hat.swapaxes(0,2)

            m = m.swapaxes(0,2)
            m = m.reshape(1,self.n_channels,-1, self.window_size*self.window_hierarchy[-1])
            m = m.swapaxes(0,2)

            if self.normalize_windows:
                x = x + x_scales
                x_hat = x_hat + x_scales
                x = x * m
                x_hat = x_hat * m

            # Loss and update
            L = 0.5 * self.z_sigma * self.z_sigma * mse(x, x_hat)
            optim.zero_grad()
            L.backward()
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 5.0)#梯度裁剪，防止连乘过多
            optim.step()
            lr_scheduler.step()
            mse_list.append(L.cpu().data.numpy())

            if self.z_persistent:
                for j in range(len(self.z_size)):
                    temp[j] = temp[j].reshape(self.batch_size,self.window_hierarchy[j],self.z_size[j],1,1)
                    self.p_0_chains[j][chains_i] = temp[j]
            if not plot_mse:
                if i % 50 == 0:
                    norm_z0 = torch.norm(z_0[0], dim=0).mean()
                    norm_z = torch.norm(z, dim=0).mean()
                    batch_size = len(x)
                    print('{:>6d} mse(x, x_hat)={:>10.4f} norm(z0)={:>10.4f} norm(z)={:>10.4f} time={:>10.4f}'.format(i, np.mean(mse_list) / batch_size,
                                                                                                                  norm_z0, norm_z,
                                                                                                                  time.time()-start_time))
                    mse_list = []
        
        if plot_mse:
            plt.plot(range(len(mse_list)),mse_list)
            # plt.savefig(f'mse-hierarchy={self.window_hierarchy}')
            plt.savefig(path + f'\mse-hierarchy={self.window_hierarchy}.png')
        self.generator.eval()
    
    def predict(self, X, mask, z_iters):#(n_window, n_features, 1, window_size*window_hierarchy size)
        self.generator.eval()
        #x:mask过后的结果
        #z_0：服从高斯分布的随机数 作为初始值 无特殊意义
        #get_z得到z：试图去寻找会导致这一输出的输入
        #x_hat:z作为输入时的输出

        # Get full batch
        x, z_0, m, _, x_scales = self.get_batch(X=X, mask=mask, batch_size=len(X), p_0_chains=[None for i in range(len(self.z_size))], z_persistent=False, shuffle=False)

        # Forward
        z, _ = self.get_z(z=z_0, x=x, m=m, n_iters=z_iters, with_noise=False)
        m = torch.ones(m.shape).to(self.device) # In forward of generator, mask is all ones to reconstruct everything
        x_hat = self.generator(z, m)
        
        # Return to window_size * window_hierarchy size
        x = x.swapaxes(0,2)
        x = x.reshape(1,self.n_channels,-1, self.window_size*self.window_hierarchy[-1])
        x = x.swapaxes(0,2)
        # (n_window, n_features, 1, window_size*window_hierarchy size)

        x_hat = x_hat.swapaxes(0,2)
        x_hat = x_hat.reshape(1,self.n_channels,-1, self.window_size*self.window_hierarchy[-1])
        x_hat = x_hat.swapaxes(0,2)

        m = m.swapaxes(0,2)
        m = m.reshape(1,self.n_channels,-1, self.window_size*self.window_hierarchy[-1])
        m = m.swapaxes(0,2)

        x = x.cpu().data.numpy()
        x_hat = x_hat.cpu().data.numpy()
        z = z.cpu().data.numpy()
        m = m.cpu().data.numpy()

        return x, x_hat, z, m

    def z_analysis(self, z_l, z_u, n_interpolation): # 仿照section4.4的分析
        total_series = []
        for z in np.linspace(z_l, z_u, n_interpolation):
            z = torch.Tensor(z[None,:,:,:])
            x = self.generator(z, None).detach()
            x = x.swapaxes(0,2)
            x = x.reshape(1,self.n_channels,-1, self.window_size)
            x = x.swapaxes(0,2)
            total_series = total_series + list(x[0, 0, 0,:]) + [None for i in range(50)]
        plt.plot(np.array(total_series))
        plt.show()
        

    def anomaly_score(self, X, mask, z_iters):
        x, x_hat, z, mask = self.predict(X=X, mask=mask, z_iters=z_iters)
        x_hat = x_hat*mask # Hide non-available data

        x_flatten = x.squeeze(2)#移除维度为1的维度
        x_hat_flatten = x_hat.squeeze(2)
        mask_flatten = mask.squeeze(2)
        #(n_window, n_features, window_size*window_hierarchy size)
        z = z.squeeze((2,3))

        ts_score = np.square(x_flatten-x_hat_flatten)
        #(n_window, n_features, window_size*window_hierarchy size)
        score = np.average(ts_score, axis=1, weights=mask_flatten)
        #(n_rwindow, window_size*window_hierarchy size)
        return score, ts_score, x, x_hat, z, mask
