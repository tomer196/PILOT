import torch
from torch import nn
from torch.nn import functional as F
from models.rec_models.unet_model import UnetModel
import pytorch_nufft.nufft as nufft
import pytorch_nufft.interp as interp
import data.transforms as transforms
from scipy.spatial import distance_matrix
from tsp_solver.greedy import solve_tsp
import numpy as np

class Subsampling_Layer(nn.Module):
    def initilaize_trajectory(self,trajectory_learning,initialization):
        x = torch.zeros(self.num_measurements, 2)
        if initialization == 'spiral':
            x=np.load(f'spiral/{self.decimation_rate}spiral.npy')
            x=torch.tensor(x).float()
        elif initialization == 'EPI':
            index = 0
            for i in range(self.res // self.decimation_rate):
                if i % 2 == 0:
                    for j in range(self.res):
                        x[index, 1] = i * self.decimation_rate + self.decimation_rate / 2 - 160
                        x[index, 0] = j - 160
                        index += 1
                else:
                    for j in range(self.res):
                        x[index, 1] = i * self.decimation_rate + self.decimation_rate / 2 - 160
                        x[index, 0] = self.res - j - 1 - 160
                        index += 1
        elif initialization == 'rossete':
            k_max = self.res // 2
            w1 = 1087
            w2 = 113
            r = torch.arange(self.num_measurements, dtype=torch.float64) * 1e-4
            x[:, 0] = k_max * torch.sin(w1 * r) * torch.sin(w2 * r)
            x[:, 1] = k_max * torch.sin(w1 * r) * torch.cos(w2 * r)
        elif initialization == 'uniform':
            x = (torch.rand(self.num_measurements, 2) - 0.5) * self.res
        elif initialization == 'gaussian':
            x = torch.randn(self.num_measurements, 2) * self.res/6
        else:
            print('Wrong initialization')
        self.x = torch.nn.Parameter(x, requires_grad=trajectory_learning)
        return


    def __init__(self, decimation_rate, res,trajectory_learning,initialization):
        super().__init__()

        self.decimation_rate=decimation_rate
        self.res=res
        self.num_measurements=res**2//decimation_rate
        self.initilaize_trajectory(trajectory_learning, initialization)

    def forward(self, input):
        input = input.permute(0, 1, 4, 2, 3).squeeze(1)
        sub_ksp = interp.bilinear_interpolate_torch_gridsample(input, self.x)
        output = nufft.nufft_adjoint(sub_ksp, self.x, input.shape)
        return output.unsqueeze(1)

    def get_trajectory(self):
        return self.x

    def __repr__(self):
        return f'Subsampling_Layer'

class Subsampling_Model(nn.Module):
    def __init__(self, in_chans, out_chans, chans, num_pool_layers, drop_prob,decimation_rate,res,trajectory_learning,initialization):
        super().__init__()

        self.subsampling=Subsampling_Layer(decimation_rate, res,trajectory_learning,initialization)
        self.reconstruction_model=UnetModel(in_chans, out_chans, chans, num_pool_layers, drop_prob)

    def forward(self, input):
        input=self.subsampling(input)
        output = self.reconstruction_model(input)
        return output

    def get_trajectory(self):
        return self.subsampling.get_trajectory()
