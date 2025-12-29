r""" 
Prior class of the orbital parameters
 for the Beta Pictoris system.
"""

import torch
import numpy as np
from torch.distributions import Uniform, Normal


import torch
import numpy as np
from torch.distributions import Uniform, Normal

class Prior:
    """
    Unified Informed Priors for Beta Pictoris b (Training/Data Gen)
    """
    def __init__(self):
        # 1. 半长轴 (SMA): [8.0, 13.0] AU
        # 范围缩小理由：已知的直接成像观测约束
        self.SMA_LOWER = torch.log(torch.tensor(8.0))
        self.SMA_UPPER = torch.log(torch.tensor(13.0))

        # 2. 偏心率 (ECC): [0.0, 0.3]
        # 范围缩小理由：Beta Pic b 被认为是低偏心率轨道
        self.ECC_LOWER = torch.tensor(0.0)
        self.ECC_UPPER = torch.tensor(0.3)

        # 3. 倾角 (INC): [85.0, 95.0] 度
        # 范围缩小理由：系统是侧向的 (Edge-on)，且有凌星迹象，非常接近 90 度
        # 注意：这里保持 cos 采样结构，对应各向同性先验的局部
        self.INC_LOWER = torch.cos(torch.tensor(np.radians(95.0))) 
        self.INC_UPPER = torch.cos(torch.tensor(np.radians(85.0))) # cos(85) > cos(95)

        # 4. 近星点幅角 (AOP): [0, 360] 度
        # 理由：对于低偏心率轨道，AOP 很难约束，必须保持全范围以防偏差
        self.AOP_LOWER = torch.tensor(0.0)
        self.AOP_UPPER = torch.tensor(360.0)

        # 5. 升交点经度 (PAN): [28.0, 38.0] 度
        # 范围缩小理由：岩屑盘 PA 约为 32 度，行星与之共面
        self.PAN_LOWER = torch.tensor(28.0)
        self.PAN_UPPER = torch.tensor(38.0)

        # 6. 过近星点时刻 (TAU): [0.0, 1.0]
        self.TAU_LOWER = torch.tensor(0.0)
        self.TAU_UPPER = torch.tensor(1.0)

        # --- 以下保持高斯分布不变 ---
        self.PLX_MEAN = torch.tensor(51.44)
        self.PLX_STD = torch.tensor(0.12)

        self.MTOT_MEAN = torch.tensor(1.75)
        self.MTOT_STD = torch.tensor(0.05)

        # Distributions
        self.sma_dist = Uniform(self.SMA_LOWER, self.SMA_UPPER)
        self.ecc_dist = Uniform(self.ECC_LOWER, self.ECC_UPPER)
        self.inc_dist = Uniform(self.INC_LOWER, self.INC_UPPER) # 注意上下界顺序，Uniform会自动处理min/max
        self.aop_dist = Uniform(self.AOP_LOWER, self.AOP_UPPER)
        self.pan_dist = Uniform(self.PAN_LOWER, self.PAN_UPPER)
        self.tau_dist = Uniform(self.TAU_LOWER, self.TAU_UPPER)
        self.plx_dist = Normal(self.PLX_MEAN, self.PLX_STD)
        self.mtot_dist = Normal(self.MTOT_MEAN, self.MTOT_STD)

        self.sma_dist = Uniform(self.SMA_LOWER, self.SMA_UPPER)
        self.ecc_dist = Uniform(self.ECC_LOWER, self.ECC_UPPER)
        self.inc_dist = Uniform(self.INC_LOWER, self.INC_UPPER)
        self.aop_dist = Uniform(self.AOP_LOWER, self.AOP_UPPER)
        self.pan_dist = Uniform(self.PAN_LOWER, self.PAN_UPPER)
        self.tau_dist = Uniform(self.TAU_LOWER, self.TAU_UPPER)
        self.plx_dist = Normal(self.PLX_MEAN, self.PLX_STD)
        self.mtot_dist = Normal(self.MTOT_MEAN, self.MTOT_STD)
        
    def sample(self, ndims):
        """
        Samples from the prior distributions for the parameters.

        Args:
            ndims (int): The number of samples to generate.

        Returns:
            torch.Tensor: A tensor of shape (ndims, 8) containing the generated samples.
        """
        sma = torch.exp(self.sma_dist.sample(ndims))
        ecc = self.ecc_dist.sample(ndims)
        inc = np.degrees(torch.acos(self.inc_dist.sample(ndims)).clone().detach())
        aop = self.aop_dist.sample(ndims)
        pan = self.pan_dist.sample(ndims)
        tau = self.tau_dist.sample(ndims)
        plx = self.plx_dist.sample(ndims)
        mtot = self.mtot_dist.sample(ndims)
        samples = torch.cat((
            sma.unsqueeze(1), 
            ecc.unsqueeze(1), 
            inc.unsqueeze(1), 
            aop.unsqueeze(1), 
            pan.unsqueeze(1), 
            tau.unsqueeze(1), 
            plx.unsqueeze(1), 
            mtot.unsqueeze(1)), dim=1)
        return samples
    
    def pre_process(self, theta):
        """
        Pre-processes the generated samples.

        Args:
            theta (torch.Tensor): A tensor of shape (ndims, 8) containing the generated samples.

        Returns:
            torch.Tensor: A tensor of shape (ndims, 8) containing the pre-processed samples.
        """
        sma, ecc, inc, aop, pan, tau, plx, mtot = theta[:, 0], theta[:, 1], theta[:, 2], theta[:, 3], theta[:, 4], theta[:, 5], theta[:, 6], theta[:, 7]

        sma = 2 * (torch.log(sma) - self.SMA_LOWER) / (self.SMA_UPPER - self.SMA_LOWER) - 1
        ecc = 2 * (ecc - self.ECC_LOWER) / (self.ECC_UPPER - self.ECC_LOWER) - 1
        inc = 2 * (torch.cos(np.radians(inc)) - self.INC_LOWER) / (self.INC_UPPER - self.INC_LOWER) - 1
        aop = 2 * (aop - self.AOP_LOWER) / (self.AOP_UPPER - self.AOP_LOWER) - 1
        pan = 2 * (pan - self.PAN_LOWER) / (self.PAN_UPPER - self.PAN_LOWER) - 1
        tau = 2 * (tau - self.TAU_LOWER) / (self.TAU_UPPER - self.TAU_LOWER) - 1
        plx = (plx - self.PLX_MEAN) / (self.PLX_STD*3) # 3 times the standard deviation to get 99.7% of the data between -1 and 1
        mtot = (mtot - self.MTOT_MEAN) / (self.MTOT_STD*3) 

        theta = torch.cat((
            sma.unsqueeze(1), 
            ecc.unsqueeze(1), 
            inc.unsqueeze(1), 
            aop.unsqueeze(1), 
            pan.unsqueeze(1), 
            tau.unsqueeze(1), 
            plx.unsqueeze(1), 
            mtot.unsqueeze(1)), dim=1)

        return theta
    
    def post_process(self, theta):
        """
        Post-processes the generated samples.

        Args:
            theta (torch.Tensor): A tensor of shape (ndims, 8) containing the generated samples.

        Returns:
            torch.Tensor: A tensor of shape (ndims, 8) containing the post-processed samples.
        """
        sma, ecc, inc, aop, pan, tau, plx, mtot = theta[:, 0], theta[:, 1], theta[:, 2], theta[:, 3], theta[:, 4], theta[:, 5], theta[:, 6], theta[:, 7]

        sma = torch.exp((sma + 1) * (self.SMA_UPPER - self.SMA_LOWER) / 2 + self.SMA_LOWER)
        ecc = (ecc + 1) * (self.ECC_UPPER - self.ECC_LOWER) / 2 + self.ECC_LOWER
        inc = np.degrees(torch.acos((inc + 1) * (self.INC_UPPER - self.INC_LOWER) / 2 + self.INC_LOWER))
        aop = (aop + 1) * (self.AOP_UPPER - self.AOP_LOWER) / 2 + self.AOP_LOWER
        pan = (pan + 1) * (self.PAN_UPPER - self.PAN_LOWER) / 2 + self.PAN_LOWER
        tau = (tau + 1) * (self.TAU_UPPER - self.TAU_LOWER) / 2 + self.TAU_LOWER
        plx = plx * self.PLX_STD*3 + self.PLX_MEAN
        mtot = mtot * self.MTOT_STD*3 + self.MTOT_MEAN

        theta = torch.cat((
            sma.unsqueeze(1), 
            ecc.unsqueeze(1), 
            inc.unsqueeze(1), 
            aop.unsqueeze(1), 
            pan.unsqueeze(1), 
            tau.unsqueeze(1), 
            plx.unsqueeze(1), 
            mtot.unsqueeze(1)), dim=1)

        return theta