import numpy as np
import orbitize
from orbitize import read_input, system, priors, sampler
import matplotlib.pyplot as plt

def main():
    data_table = read_input.read_file('{}/betaPic.csv'.format(orbitize.DATADIR))
    data_table = data_table[:-1] # Discard the RV observation, as we do not take it into account in the model

    num_planets = 1
    total_mass = 1.75 # [Msol]
    plx = 51.44 # [mas]
    mass_err = 0.05 # [Msol]
    plx_err = 0.12 # [mas]

    sys = system.System(
        num_planets, data_table, total_mass,
        plx, mass_err=mass_err, plx_err=plx_err
    )

    lab = sys.param_idx

    # set up the same priors as for https://arxiv.org/abs/2201.08506v1(github : https://github.com/HeSunPU/DPI/tree/main)
# 1. SMA: 8.0 - 13.0 AU
    sys.sys_priors[lab['sma1']] = priors.UniformPrior(8.0, 13.0)

    # 2. ECC: 0.0 - 0.3
    sys.sys_priors[lab['ecc1']] = priors.UniformPrior(0.0, 0.3)

    # 3. INC: 85 - 95 度 (转换为弧度)
    sys.sys_priors[lab['inc1']] = priors.UniformPrior(np.deg2rad(85.0), np.deg2rad(95.0))

    # 4. AOP: 0 - 360 度 (保持全范围，因为圆轨道简并)
    sys.sys_priors[lab['aop1']] = priors.UniformPrior(0.0, 2*np.pi)

    # 5. PAN: 28 - 38 度 (紧紧围绕观测到的盘面角度)
    sys.sys_priors[lab['pan1']] = priors.UniformPrior(np.deg2rad(28.0), np.deg2rad(38.0))

    # 6. TAU: 0 - 1
    sys.sys_priors[lab['tau1']] = priors.UniformPrior(0.0, 1.0)

    # number of temperatures & walkers for MCMC
    num_temps = 20
    num_walkers = 1000

    # number of steps to take
    n_orbs = 10000000

    mcmc_sampler = sampler.MCMC(sys, num_temps, num_walkers, num_threads=32)

    _ = mcmc_sampler.run_sampler(n_orbs, output_filename='mcmc_betapic.hdf5')

if __name__ == '__main__':
    main()