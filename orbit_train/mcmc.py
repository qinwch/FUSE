import argparse
from pathlib import Path

from path_config import DEFAULT_ORBIT_PATHS


def main(output, num_temps, num_walkers, n_orbs, num_threads):
    import numpy as np
    import orbitize
    from orbitize import read_input, system, priors, sampler

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

    mcmc_sampler = sampler.MCMC(sys, num_temps, num_walkers, num_threads=num_threads)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _ = mcmc_sampler.run_sampler(n_orbs, output_filename=str(output_path))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the betapic PTMCMC reference sampler")
    parser.add_argument("--output", type=str, default=str(DEFAULT_ORBIT_PATHS.reference_path("mcmc_betapic.hdf5")), help="Output HDF5 filename")
    parser.add_argument("--num-temps", type=int, default=20, help="Number of MCMC temperatures")
    parser.add_argument("--num-walkers", type=int, default=1000, help="Number of MCMC walkers")
    parser.add_argument("--n-orbs", type=int, default=10000000, help="Number of MCMC steps")
    parser.add_argument("--num-threads", type=int, default=32, help="Number of sampler threads")
    args = parser.parse_args()

    main(args.output, args.num_temps, args.num_walkers, args.n_orbs, args.num_threads)
