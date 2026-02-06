r""" 
Prior class of the orbital parameters
 for the Beta Pictoris system.
"""

import torch
import numpy as np
from torch.distributions import Uniform, Normal


class Prior:
    """
    Class for defining the prior distributions for the parameters used for
    calculating an orbit for Beta Pictoris b.
    """
    def __init__(self):
        """
        Initializes the prior distributions for the parameters.
        """
        
        self.SMA_LOWER = torch.log(torch.tensor(4.0))
        self.SMA_UPPER = torch.log(torch.tensor(40.0))

        self.ECC_LOWER = torch.tensor(1e-5)
        self.ECC_UPPER = torch.tensor(0.99)

        self.INC_LOWER = torch.cos(torch.tensor(np.radians(99.0)))
        self.INC_UPPER = torch.cos(torch.tensor(np.radians(81.0)))

        self.AOP_LOWER = torch.tensor(0.0)
        self.AOP_UPPER = torch.tensor(360.0)

        self.PAN_LOWER = torch.tensor(25.0)
        self.PAN_UPPER = torch.tensor(85.0)

        self.TAU_LOWER = torch.tensor(0.0)
        self.TAU_UPPER = torch.tensor(1.0)

        self.PLX_MEAN = torch.tensor(51.44)
        self.PLX_STD = torch.tensor(0.12)

        self.MTOT_MEAN = torch.tensor(1.75)
        self.MTOT_STD = torch.tensor(0.05)

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
    
    def log_prob(self, theta):
        """
        Calculates the log probability of the physical parameters.

        Args:
            theta (torch.Tensor): A tensor of shape (ndims, 8) containing the physical samples.
                                  Order: sma, ecc, inc, aop, pan, tau, plx, mtot

        Returns:
            torch.Tensor: A tensor of shape (ndims, 1) containing the sum of log probabilities.
        """
        # Ensure constants are on the same device as input
        device = theta.device
        
        # Unpack parameters
        sma = theta[:, 0]
        ecc = theta[:, 1]
        inc = theta[:, 2]
        aop = theta[:, 3]
        pan = theta[:, 4]
        tau = theta[:, 5]
        plx = theta[:, 6]
        mtot = theta[:, 7]

        # 1. SMA: Defined as Uniform on Log(SMA)
        # Distribution is on y = log(x). We have x.
        # p(x) = p(y) * |dy/dx| = p(log x) * (1/x)
        # log p(x) = log p(log x) - log(x)
        log_sma = torch.log(sma)
        lp_sma = self.sma_dist.log_prob(log_sma) - log_sma

        # 2. ECC: Uniform
        lp_ecc = self.ecc_dist.log_prob(ecc)

        # 3. INC: Defined as Uniform on Cos(INC_rad)
        # Distribution is on y = cos(x_rad). We have x_deg.
        # x_rad = x_deg * pi / 180
        # y = cos(x_deg * pi / 180)
        # dy/dx_deg = -sin(x_rad) * (pi / 180)
        # p(x_deg) = p(y) * |dy/dx_deg|
        # log p(x_deg) = log p(y) + log(sin(x_rad)) + log(pi/180)
        inc_rad = torch.deg2rad(inc)
        cos_inc = torch.cos(inc_rad)
        # Note: We take absolute value of Jacobian, but sin(inc) is positive in [0, 180]
        jacobian_inc = torch.log(torch.sin(inc_rad)) + torch.tensor(np.log(np.pi/180.0), device=device)
        lp_inc = self.inc_dist.log_prob(cos_inc) + jacobian_inc

        # 4. AOP: Uniform
        lp_aop = self.aop_dist.log_prob(aop)

        # 5. PAN: Uniform
        lp_pan = self.pan_dist.log_prob(pan)

        # 6. TAU: Uniform
        lp_tau = self.tau_dist.log_prob(tau)

        # 7. PLX: Normal
        lp_plx = self.plx_dist.log_prob(plx)

        # 8. MTOT: Normal
        lp_mtot = self.mtot_dist.log_prob(mtot)

        # Sum up log probabilities for all dimensions
        # If any value is out of bounds, the corresponding lp_ will be -inf, making the sum -inf.
        log_prior = lp_sma + lp_ecc + lp_inc + lp_aop + lp_pan + lp_tau + lp_plx + lp_mtot
        
        return log_prior.unsqueeze(1)
