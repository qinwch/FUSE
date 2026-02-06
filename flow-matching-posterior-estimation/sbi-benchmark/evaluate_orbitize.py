"""
Beta Pic b orbit inference + FMPE sampling (refactored for readability)

Changes:
- All imports are consolidated at the top
- All function definitions are grouped together
- Main execution is organized into a clear pipeline
"""

# =========================
# Imports
# =========================
import math
import os
from os.path import join

import h5py
import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
from matplotlib.offsetbox import HPacker, TextArea

import orbitize
from orbitize import read_input, DATADIR
from orbitize import priors as P
from orbitize import system, sampler
from orbitize.system import seppa2radec, transform_errors

from lampe.plots import corner

from dingo.core.posterior_models.build_model import build_model_from_kwargs

# Local utilities
from helpers import Simulator, Prior


# =========================
# Functions
# =========================
def load_beta_pic_table(drop_rv: bool = True):
    """Load betaPic.csv from orbitize DATADIR. Optionally drop the last RV row."""
    data_table = read_input.read_file(f"{DATADIR}/betaPic.csv")
    if drop_rv:
        data_table = data_table[:-1]
    return data_table


def convert_seppa_table_to_radec(data_table):
    """
    Convert (sep, pa) astrometry table to (RA, Dec) with transformed errors/correlation,
    and set quant_type to 'radec'.
    """
    ra, dec = seppa2radec(data_table["quant1"], data_table["quant2"])

    ra_errs, dec_errs, corrs = [], [], []
    for i in range(len(data_table)):
        ra_err, dec_err, corr = transform_errors(
            data_table["quant1"][i],
            data_table["quant2"][i],
            data_table["quant1_err"][i],
            data_table["quant2_err"][i],
            data_table["quant12_corr"][i],
            seppa2radec,
        )
        ra_errs.append(ra_err)
        dec_errs.append(dec_err)
        corrs.append(corr)

    data_table["quant1"] = ra
    data_table["quant2"] = dec
    data_table["quant1_err"] = ra_errs
    data_table["quant2_err"] = dec_errs
    data_table["quant12_corr"] = corrs
    data_table["quant_type"] = ["radec"] * len(data_table)

    return data_table


def build_orbitize_system_and_sampler(
    data_table,
    num_planets=1,
    total_mass=1.75,
    plx=51.44,
    mass_err=0.05,
    plx_err=0.12,
    num_temps=10,
    num_walkers=100,
):
    """Create orbitize System, set priors, and create an MCMC sampler."""
    sys = system.System(
        num_planets,
        data_table,
        total_mass,
        plx,
        mass_err=mass_err,
        plx_err=plx_err,
    )

    lab = sys.param_idx

    # Priors (same as https://arxiv.org/abs/2201.08506v1 mentioned in your comment)
    sys.sys_priors[lab["sma1"]] = P.UniformPrior(4.0, 40.0)
    sys.sys_priors[lab["ecc1"]] = P.UniformPrior(0.00001, 0.99)
    sys.sys_priors[lab["inc1"]] = P.UniformPrior(np.deg2rad(81), np.deg2rad(99))
    sys.sys_priors[lab["aop1"]] = P.UniformPrior(0, 2 * np.pi)
    sys.sys_priors[lab["pan1"]] = P.UniformPrior(np.deg2rad(25), np.deg2rad(85))
    sys.sys_priors[lab["tau1"]] = P.UniformPrior(0, 1)

    mcmc_sampler = sampler.MCMC(sys, num_temps=num_temps, num_walkers=num_walkers, num_threads=1)
    return sys, mcmc_sampler


def build_x_star_from_table(data_table, scale=1e6, dtype=torch.float32):
    """
    Build x_star = [2*T] tensor interleaving (ra, dec), then scale.
    Assumes data_table["quant1"]=RA, data_table["quant2"]=Dec.
    """
    ra_obs = torch.tensor(data_table["quant1"].data, dtype=dtype)
    dec_obs = torch.tensor(data_table["quant2"].data, dtype=dtype)

    x_star = torch.empty(2 * len(data_table), dtype=dtype)
    x_star[0::2] = ra_obs
    x_star[1::2] = dec_obs
    x_star = x_star / scale
    return x_star


def load_fmpe_model(train_dir: str):
    """Load settings + model checkpoint from a training directory."""
    with open(join(train_dir, "settings.yaml"), "r") as f:
        settings = yaml.safe_load(f)

    # Prefer latest checkpoint (adjust if you want best_model.pt etc.)
    ckpt = join(train_dir, "model_latest.pt")

    model = build_model_from_kwargs(
        filename=ckpt,
        device=settings["training"].get("device", "cpu"),
    )
    return model, settings


def get_orbitize_scores(theta_pred_tensor, orbitize_prior: Prior, orbitize_sampler):
    """
    Compute orbitize (log-likelihood + log-prior) score for a batch of theta.

    Return:
        torch.Tensor shape [B]
    """
    B = theta_pred_tensor.shape[0]
    device = theta_pred_tensor.device

    # Postprocess to physical params, then orbitize expects radians for angles
    phys = (
        orbitize_prior.post_process(theta_pred_tensor.cpu())
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )

    # If post_process outputs degrees for angles, convert to radians for orbitize:
    phys[:, 2] = np.deg2rad(phys[:, 2])  # inc
    phys[:, 3] = np.deg2rad(phys[:, 3])  # aop
    phys[:, 4] = np.deg2rad(phys[:, 4])  # pan

    # 1) log-likelihood only
    logl = orbitize_sampler._logl(phys, include_logp=False)
    logl = np.asarray(logl).reshape(-1)  # [B]

    # 2) log-prior
    logp = np.zeros(B, dtype=np.float64)
    for j, prior in enumerate(orbitize_sampler.priors):
        logp += np.asarray(prior.compute_lnprob(phys[:, j])).reshape(-1)

    scores = logl + logp
    scores[~np.isfinite(scores)] = -np.inf
    return torch.from_numpy(scores).to(device=device, dtype=torch.float32)


def sample_with_guidance(
    model,
    x_star: torch.Tensor,
    score_fn,
    num_target_samples=4096,
    n_batches=1,
):
    """Sample posterior with external score guidance in batches."""
    batch_size = math.ceil(num_target_samples / n_batches)
    posterior_samples_list = []
    t1_avg_list, t1_std_list = [], []

    print(f"Sampling {num_target_samples} samples in {n_batches} batches (batch size: {batch_size})...")

    for i in range(n_batches):
        batch_samples, t1_avg_dict, t1_std_dict = model.sample(
            x_star,
            num_samples=batch_size,
            _score_fn=score_fn,
        )
        posterior_samples_list.append(batch_samples)
        t1_avg_list.append(t1_avg_dict)
        t1_std_list.append(t1_std_dict)

    posterior_samples_norm = torch.cat(posterior_samples_list, dim=0)[:num_target_samples]
    return posterior_samples_norm, t1_avg_list, t1_std_list

def sample_and_log_prob_with_guidance(
    model,
    x_star: torch.Tensor,
    score_fn,
    num_target_samples=4096,
    n_batches=1,
):
    """Sample posterior with external score guidance in batches."""
    batch_size = math.ceil(num_target_samples / n_batches)
    posterior_samples_list = []
    log_prob_list = []
    t1_avg_list, t1_std_list = [], []

    print(f"Sampling {num_target_samples} samples in {n_batches} batches (batch size: {batch_size})...")

    for i in range(n_batches):
        batch_samples, _, _, log_prob = model.sample_and_log_prob(
            x_star,
            num_samples=batch_size,
            _score_fn=score_fn,
        )
        posterior_samples_list.append(batch_samples)
        log_prob_list.append(log_prob)

    posterior_samples_norm = torch.cat(posterior_samples_list, dim=0)[:num_target_samples]
    log_prob_final = torch.cat(log_prob_list, dim=0)[:num_target_samples]
    return posterior_samples_norm, log_prob_final

def load_mcmc_samples_hdf5(filename: str, burn_in_per_chain=100, n_chains=1000, chain_len=1000, dim=8):
    """
    Load MCMC samples from HDF5 and reshape:
      post: (n_chains * chain_len, dim) -> (n_chains, chain_len, dim)
    Then remove burn-in and flatten.
    """
    with h5py.File(filename, "r") as hf:
        post = np.array(hf.get("post"))

    samples = post.reshape((n_chains, chain_len, dim))
    samples = samples[:, burn_in_per_chain:, :]
    samples = samples.reshape([-1, dim])

    # Convert angles rad -> deg for comparison with NPE (if NPE uses degrees)
    samples[:, 2] = np.rad2deg(samples[:, 2])
    samples[:, 3] = np.rad2deg(samples[:, 3])
    samples[:, 4] = np.rad2deg(samples[:, 4])
    return samples


def corner_plot(npe_samples, mcmc_samples, train_dir: str, filename: str = "corner.png"):
    """Overlay corner plot of NPE samples and MCMC samples, then save to train_dir."""
    LOWER = torch.tensor([8.0, 0.0, 85.5, 0.0, 29.6, 0.0, 50.8, 1.5])
    UPPER = torch.tensor([40.0, 0.9, 93.0, 360.0, 33.0, 1.0, 52.1, 2.0])

    LABELS = [
        r"$a$", r"$e$", r"$i$",
        r"$\omega$", r"$\Omega$",
        r"$\tau$", r"$\pi$", r"$M_T$",
    ]

    plt.rcParams["text.usetex"] = False

    fig = corner(
        npe_samples,
        domain=(LOWER, UPPER),
        bins=64,
        labels=LABELS,
    )

    corner(
        mcmc_samples,
        domain=(LOWER, UPPER),
        bins=64,
        labels=LABELS,
        legend=r"MCMC",
        figure=fig,
    )

    os.makedirs(train_dir, exist_ok=True)
    outpath = join(train_dir, filename)

    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return outpath


# =========================
# Main
# =========================
def main():
    # (Optional) set the directory to the root of the project
    # os.chdir("../")

    # 1) Load and convert observation table to RA/Dec
    data_table = load_beta_pic_table(drop_rv=True)
    data_table = convert_seppa_table_to_radec(data_table)

    # 2) Build local Prior / Simulator (used by your model + post_process)
    priors = Prior()
    simulator = Simulator(data_table)  # kept for completeness (even if not used below)

    # 3) Build orbitize system + sampler for scoring
    _, mcmc_sampler = build_orbitize_system_and_sampler(data_table)

    # 4) Build x_star (context) for the FMPE model
    x_star = build_x_star_from_table(data_table, scale=1e6)

    # 5) Load FMPE model
    train_dir = "/102437/qinwch/flow_matching_record/orbitize_training"
    model, settings = load_fmpe_model(train_dir)

    # 6) Score function closure (guidance)
    def score_fn_closure(theta_pred_tensor):
        return get_orbitize_scores(
            theta_pred_tensor,
            orbitize_prior=priors,
            orbitize_sampler=mcmc_sampler,
        )

    # 7) Sample posterior with guidance
    num_target_samples = 4096
    n_batches = 1

    if x_star.ndim == 1:
        x_star = x_star.unsqueeze(0)
    # x_star = x_star.repeat((num_target_samples,1))
    print(f"x_star:{x_star.shape}")
    import time
    time_start = time.time()
    posterior_samples_norm, log_prob = sample_and_log_prob_with_guidance(
        model=model,
        x_star=x_star,
        num_target_samples=num_target_samples,
        n_batches=n_batches,
        score_fn=score_fn_closure
    )
    print(time.time() - time_start)
    return
    print(f"posterior_samples_norm:{posterior_samples_norm.shape}")


    # 8) Postprocess to physical parameter space (for plotting)
    postprocessed_samples = priors.post_process(posterior_samples_norm.cpu()).numpy()
    save_path = join(train_dir, f"sample_soft_new_fk_1_8_new_all_03_5.npy")
    np.save(save_path, postprocessed_samples)
    print(f"Saved posterior samples to: {save_path}")
    save_path = join(train_dir, f"logp_soft_new_fk_1_8_new_all_03_5.npy")
    np.save(save_path, log_prob.cpu().numpy())
    print(f"Saved logp to: {save_path}")
    # 9) Load reference MCMC samples
    mcmc_file = "/102437/qinwch/mcmc_betapic.hdf5"
    mcmc_samples = load_mcmc_samples_hdf5(mcmc_file)

    # 10) Compare via corner plot
    outpath = corner_plot(postprocessed_samples, mcmc_samples, train_dir=train_dir, filename="corner_soft_new_fk_1_8_new_all_03_5.png")
    print(f"Saved corner plot to: {outpath}")


if __name__ == "__main__":
    main()
