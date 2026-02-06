
import argparse
import csv
import math
from os.path import join
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sbibm.tasks
from sbibm.metrics import c2st, mmd, median_distance, ksd, posterior_mean_error, posterior_variance_ratio
import torch
from torch.utils.data import Dataset
import torch.distributions as pdist
from torchdiffeq import odeint
import numpy as np
import pandas as pd
import concurrent.futures
import yaml
import wandb
# from joblib import Parallel, delayed

from dingo.core.posterior_models.build_model import (
    build_model_from_kwargs,
    autocomplete_model_kwargs,
)
from dingo.core.utils import build_train_and_test_loaders, RuntimeLimits

def compute_kl_divergence_knn(sample_p, sample_q, k=5):
    """
    使用 k-NN 估计两组样本间的 KL 散度: KL(P || Q)
    sample_p: Reference samples (真值) [N, D]
    sample_q: Posterior samples (预测值) [M, D]
    """
    # 确保在同一设备
    sample_p = sample_p.to(sample_q.device)
    
    n, d = sample_p.shape
    m, _ = sample_q.shape
    
    # 1. 计算 P 内部的 k-NN 距离 (r_k)
    # cdist 计算成对欧氏距离
    p_p_dist = torch.cdist(sample_p, sample_p, p=2) 
    # 添加一个无穷大对角线，防止把自己当做最近邻
    p_p_dist.fill_diagonal_(float('inf'))
    # 获取第 k 个最近邻的距离
    r_k, _ = torch.topk(p_p_dist, k=k, dim=1, largest=False)
    r_k = r_k[:, -1] # 取第 k 个值

    # 2. 计算 P 到 Q 的 k-NN 距离 (s_k)
    p_q_dist = torch.cdist(sample_p, sample_q, p=2)
    s_k, _ = torch.topk(p_q_dist, k=k, dim=1, largest=False)
    s_k = s_k[:, -1]

    # 3. 应用 k-NN KL 估计公式
    # KL approx = (d/n) * sum(log(s_k/r_k)) + log(m/(n-1))
    # 为了数值稳定性，加一个极小值 eps
    eps = 1e-10
    log_ratio = torch.log(s_k + eps) - torch.log(r_k + eps)
    kl = (d / n) * torch.sum(log_ratio) + math.log(m / (n - 1))
    
    return max(0.0, kl.item()) # KL 理论上 >= 0

def compute_sinkhorn_distance(x, y, epsilon=0.1, n_iters=50):
    """
    计算 Sinkhorn 距离 (近似 Wasserstein 距离)
    x: [N, D]
    y: [M, D]
    """
    x = x.to(y.device)
    n = x.shape[0]
    m = y.shape[0]
    d = x.shape[1]

    # 计算成本矩阵 (平方欧氏距离)
    # C size: [N, M]
    C = torch.cdist(x, y, p=2) ** 2 
    
    # 归一化成本矩阵以提高稳定性
    C = C / C.max()

    # 初始化对偶向量
    mu = torch.empty(n, dtype=x.dtype, device=x.device).fill_(1.0 / n)
    nu = torch.empty(m, dtype=x.dtype, device=x.device).fill_(1.0 / m)
    
    u = torch.zeros_like(mu)
    v = torch.zeros_like(nu)

    # Sinkhorn 迭代 (Log-domain 更加稳定)
    K_log = -C / epsilon
    
    for _ in range(n_iters):
        # u = log(mu) - logsumexp(K_log + v)
        u = torch.log(mu) - torch.logsumexp(K_log + v.unsqueeze(0), dim=1)
        # v = log(nu) - logsumexp(K_log.T + u)
        v = torch.log(nu) - torch.logsumexp(K_log.transpose(0, 1) + u.unsqueeze(0), dim=1)

    # 计算传输计划 P = exp(u + K_log + v)
    # Distance = sum(P * C)
    # 由于我们在 Log 域，利用对偶公式计算距离下界通常更快且够用
    # Sinkhorn Distance ≈ <u, mu> + <v, nu>
    # 但为了直观对应 Cost，我们计算 sum(exp(coupling) * C)
    
    P_log = u.unsqueeze(1) + K_log + v.unsqueeze(0)
    P = torch.exp(P_log)
    dist = torch.sum(P * C)
    
    return dist.item()

class SbiDataset(Dataset):
    def __init__(self, theta, x):
        super(SbiDataset, self).__init__()

        self.standardization = {
            "x": {"mean": torch.mean(x, dim=0), "std": torch.std(x, dim=0)},
            "theta": {"mean": torch.mean(theta, dim=0), "std": torch.std(theta, dim=0)},
        }
        self.theta = self.standardize(theta, "theta")
        self.x = self.standardize(x, "x")

    def standardize(self, sample, label, inverse=False):
        mean = self.standardization[label]["mean"]
        std = self.standardization[label]["std"]
        sample = sample.to(mean.device)
        # print(mean.device, std.device, sample.device)

        if not inverse:
            return (sample - mean) / std
        else:
            return sample * std + mean

    def __len__(self):
        return len(self.theta)

    def __getitem__(self, idx):
        return self.theta[idx], self.x[idx]

def generate_dataset(settings, batch_size=1, directory_save=None):
    """
    Generate dataset for the given SBI benchmark task by the package sbibm.
    """

    task = sbibm.get_task(settings["task"]["name"])
    prior = task.get_prior()
    simulator = task.get_simulator()    
    num_train_samples = settings["task"]["num_train_samples"]
    nr_batches = math.ceil(num_train_samples / batch_size)
    theta = []
    x = []
    for _ in range(nr_batches):
        theta_sample = prior(batch_size)
        x_sample = simulator(theta_sample)
        theta.append(theta_sample)
        x.append(x_sample)

    # print(f"Generating {nr_batches} batches in parallel...")
    # results = Parallel(n_jobs=24, verbose=10)(
    #     delayed(generate_single_batch)(task, batch_size) for _ in range(nr_batches)
    # )

    # theta_list, x_list = zip(*results)

    x = np.vstack(x)[:num_train_samples]
    theta = np.vstack(theta)[:num_train_samples]
    if directory_save is not None:
        np.save(join(directory_save, 'x.npy'), x)
        np.save(join(directory_save, 'theta.npy'), theta)

    x = torch.tensor(x, dtype=torch.float)
    theta = torch.tensor(theta, dtype=torch.float)
    settings["task"]["dim_theta"] = theta.shape[1]
    settings["task"]["dim_x"] = x.shape[1]

    dataset = SbiDataset(theta, x)
    return dataset

def load_dataset(directory_save, settings):
    x = np.load(join(directory_save, 'x.npy'))
    theta = np.load(join(directory_save, 'theta.npy'))
    num_train_samples = settings["task"]["num_train_samples"]

    x = x[:num_train_samples]
    x = torch.tensor(x, dtype=torch.float)
    theta = theta[:num_train_samples]
    theta = torch.tensor(theta, dtype=torch.float)
    settings["task"]["dim_theta"] = theta.shape[1]
    settings["task"]["dim_x"] = x.shape[1]

    dataset = SbiDataset(theta, x)
    return dataset

def train_model(train_dir, settings, train_loader, test_loader, use_wandb=False):
    autocomplete_model_kwargs(
        settings["model"],
        input_dim=settings["task"]["dim_theta"],  # input = theta dimension
        context_dim=settings["task"]["dim_x"],  # context dim = observation dimension
    )

    model = build_model_from_kwargs(
        settings={"train_settings": settings},
        device=settings["training"].get("device", "cpu"),
    )

    # print(settings["training"].get("device", "cpu"))

    # Before training you need to call the following lines:
    model.optimizer_kwargs = settings["training"]["optimizer"]
    model.scheduler_kwargs = settings["training"]["scheduler"]
    model.initialize_optimizer_and_scheduler()

    # train model
    runtime_limits = RuntimeLimits(
        epoch_start=0,
        max_epochs_total=settings["training"]["epochs"],
    )
    print(f"early_stopping: {settings['training'].get('early_stopping', False)}")
    model.train(
        train_loader,
        test_loader,
        train_dir=train_dir,
        runtime_limits=runtime_limits,
        early_stopping=settings["training"].get("early_stopping", False),
        use_wandb=use_wandb,
    )

    return model

    # load the best model
    best_model = build_model_from_kwargs(
        filename=join(train_dir, "best_model.pt"),
        device=settings["training"].get("device", "cpu"),
    )
    return best_model

def evaluate_model(train_dir, settings, dataset, model, use_wandb=False):
    task = sbibm.get_task(settings["task"]["name"])

    is_gpu = torch.cuda.is_available()
    device = torch.device("cuda" if is_gpu else "cpu")

    # Normal test
    c2st_scores = {}
    mmd_scores = {}
    ksd_scores = {}        
    pme_scores = {}        
    pvr_scores = {}        
    meddist_scores = {}  
    kl_scores = {}       
    sinkhorn_scores = {} 
    infer_time = {}

    average_sampling_time = 0.0

    for obs in range(1, 11):
        reference_samples = task.get_reference_posterior_samples(num_observation=obs)
        reference_samples = reference_samples.to(device)
        num_samples = len(reference_samples)

        observation = dataset.standardize(
            task.get_observation(num_observation=obs), label="x"
        )
        # generate (num_samples * 2), to account for samples outside of the prior
        start_time = time.time()
        posterior_samples = model.sample_batch(observation.repeat((num_samples * 2, 1)).to(device))
        sampling_time = time.time() - start_time
        infer_time[f"infer_time_{obs}"] = sampling_time

        posterior_samples = dataset.standardize(
            posterior_samples, label="theta", inverse=True
        )

        print(f"Sampling time for observation {obs}: {sampling_time:.4f} seconds")
        average_sampling_time += sampling_time / 10

        prior_mask = torch.isfinite(task.prior_dist.log_prob(posterior_samples))
        print(
            f"{(1 - torch.sum(prior_mask) / len(prior_mask)) * 100:.2f}% of the samples "
            f"lie outside of the prior. Discarding these."
        )
        posterior_samples = posterior_samples[prior_mask].detach()
        posterior_samples = posterior_samples.to(device).detach()

        n = min(len(reference_samples), len(posterior_samples))
        post_samples_n = posterior_samples[:n]
        ref_samples_n = reference_samples[:n]

        # C2ST
        try:
            c2st_score = c2st(post_samples_n, ref_samples_n)
            c2st_scores[f"C2ST_{obs}"] = c2st_score.item()
        except Exception as e:
            print(f"Error computing C2ST: {e}")

        # MMD
        try:
            score_mmd = mmd(post_samples_n, ref_samples_n).item()
            mmd_scores[f"MMD_{obs}"] = score_mmd
        except Exception as e:
            print(f"Error computing MMD: {e}")

        # # KSD
        # try:
        #     score_ksd = ksd(task, obs, post_samples_n) 
        #     if isinstance(score_ksd, torch.Tensor):
        #         score_ksd = score_ksd.item()
        #     ksd_scores[f"KSD_{obs}"] = score_ksd
        # except Exception as e:
        #     print(f"Error computing KSD: {e}")
        #     ksd_scores[f"KSD_{obs}"] = float('nan')

        # Median Distance
        try:
            score_meddist = median_distance(post_samples_n, ref_samples_n).item()
            meddist_scores[f"MEDDIST_{obs}"] = score_meddist
        except Exception as e:
            print(f"Error computing MEDDIST: {e}")

        # Posterior Mean Error
        try:
            score_pme = posterior_mean_error(post_samples_n, ref_samples_n).item()
            pme_scores[f"PME_{obs}"] = score_pme
        except Exception as e:
            print(f"Error computing PME: {e}")

        # Posterior Variance Ratio
        try:
            score_pvr = posterior_variance_ratio(post_samples_n, ref_samples_n).item()
            pvr_scores[f"PVR_{obs}"] = score_pvr
        except Exception as e:
            print(f"Error computing PVR: {e}")

        # KL Divergence (k-NN Estimation)
        try:
            kl_n = min(n, 2000) 
            score_kl = compute_kl_divergence_knn(ref_samples_n[:kl_n], post_samples_n[:kl_n], k=5)
            kl_scores[f"KL_{obs}"] = score_kl
        except Exception as e:
            print(f"Error computing KL: {e}")
            kl_scores[f"KL_{obs}"] = float('nan')

        # Sinkhorn Distance
        try:
            # 同样限制样本数量
            sh_n = min(n, 2000)
            score_sinkhorn = compute_sinkhorn_distance(post_samples_n[:sh_n], ref_samples_n[:sh_n])
            sinkhorn_scores[f"Sinkhorn_{obs}"] = score_sinkhorn
        except Exception as e:
            print(f"Error computing Sinkhorn: {e}")
            sinkhorn_scores[f"Sinkhorn_{obs}"] = float('nan')

        fig = plt.figure(figsize=(10, 10))
        posterior_samples = posterior_samples.cpu().numpy()
        plt.scatter(
            posterior_samples[:, 0],
            posterior_samples[:, 1],
            s=0.5,
            alpha=0.2,
            label="flow matching",
        )
        reference_samples = reference_samples.cpu().numpy()
        plt.scatter(
            reference_samples[:, 0],
            reference_samples[:, 1],
            s=0.5,
            alpha=0.2,
            label="reference",
        )
        plt.title(f"C2ST: {c2st_score.item():.3f}")
        plt.legend()
        plt.savefig(join(train_dir, f"posterior_{obs}.png"))
        plt.close(fig)

    def save_csv(filename, data_dict):
        if not data_dict: return
        with open(join(train_dir, filename), "w") as f:
            w = csv.DictWriter(f, data_dict.keys())
            w.writeheader()
            w.writerow(data_dict)

    save_csv("c2st.csv", c2st_scores)
    save_csv("mmd.csv", mmd_scores)
    # save_csv("ksd.csv", ksd_scores)
    save_csv("pme.csv", pme_scores)
    save_csv("pvr.csv", pvr_scores)
    save_csv("meddist.csv", meddist_scores)
    save_csv("inference_time.csv", infer_time)
    save_csv("kl.csv", kl_scores)
    save_csv("sinkhorn.csv", sinkhorn_scores)

    if use_wandb:
        wandb.log(c2st_scores)
        wandb.log({"posteriors": wandb.Image(join(train_dir, "posteriors.png"))})

def calculate_summary_stats(x):
    """
    Extract summary statistics. 
    For Lotka-Volterra, mean and log-variance are more robust than raw time-series.
    For other tasks, identity might be sufficient.
    """
    # Assuming x is [Batch, Dim] or [Batch, Time, Dim]
    # Flatten if necessary or compute stats
    if x.ndim > 2: 
        x = x.reshape(x.shape[0], -1)
        
    # Example for general robustness: return Mean and Log-Variance
    # You can customize this based on specific tasks
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return np.concatenate([mean, np.log(var + 1e-6)], axis=-1)

def get_covariance_from_dataset(dataset):
    """
    Compute inverse covariance matrix of summary statistics from the training dataset.
    Used for Mahalanobis distance in the score function.
    """
    print("Computing covariance matrix from training dataset...")
    
    # 1. Get all training data (Standardized)
    all_x_norm = dataset.x.to("cpu")
    
    # 2. Un-standardize to Physical Space
    all_x_phys = dataset.standardize(all_x_norm, label="x", inverse=True).numpy()
    
    # 3. Calculate Stats
    stats = calculate_summary_stats(all_x_phys) # [N, Stats_Dim]

    # 4. Compute Covariance & Inverse
    cov = np.cov(stats, rowvar=False)
    # Add jitter for numerical stability
    cov = cov + np.eye(cov.shape[0]) * 1e-6
    cov_inv = np.linalg.inv(cov)
    
    return torch.from_numpy(cov_inv).float()

def get_data_weights_from_dataset(dataset):
    """
    Compute weights (inverse variance) from raw training data.
    Robust against zero variance.
    """
    print("Computing weights (inverse variance) from raw training data...")
    
    # 1. Get all training data
    all_x_norm = dataset.x.to("cpu")
    
    # 2. Un-standardize to Physical Space
    all_x_phys = dataset.standardize(all_x_norm, label="x", inverse=True).numpy()
    
    # 3. Calculate Variance
    var = np.var(all_x_phys, axis=0)
    
    # 4. Handle Zero Variance / Extreme Values
    # Replace 0 variance with 1.0 (to avoid division by zero) or a small constant
    # If variance is 0, it means this feature carries no information in the training set
    var = np.maximum(var, 1e-6) 
    
    weights = 1.0 / var
    
    # Check for NaNs/Infs
    if np.isnan(weights).any() or np.isinf(weights).any():
        print("Warning: Computed weights contain NaNs or Infs. Replacing with 1.0.")
        weights = np.nan_to_num(weights, nan=1.0, posinf=1.0, neginf=1.0)
    
    return torch.from_numpy(weights).float()

def get_sbi_scores(theta_pred_tensor, task, x_obs_stats, standardize_fn, stats_cov_inv):
    """
    Score function for SBI tasks (e.g., Lotka-Volterra).
    Score = Log_Prior - 0.5 * Mahalanobis_Distance(Sim(theta), Obs)
    """
    batch_size = theta_pred_tensor.shape[0]
    device = theta_pred_tensor.device
    
    # 1. Un-standardize Theta (Network Output -> Physical Parameters)
    theta_norm = theta_pred_tensor.detach().cpu()
    theta_phys = standardize_fn(theta_norm, label="theta", inverse=True)
    
    # 2. Prior Check (Fast Rejection)
    # If parameters are outside prior support, return -inf immediately
    log_prior = task.prior_dist.log_prob(theta_phys)
    
    # 3. Filter Valid Samples for Simulation
    valid_indices = torch.isfinite(log_prior)
    valid_theta_phys = theta_phys[valid_indices]
    
    # Initialize scores with -inf
    scores = torch.full((batch_size,), -float('inf'), dtype=torch.float32)
    
    if valid_theta_phys.shape[0] > 0:
        try:
            # 4. Run Simulator (Expensive Step)
            simulator = task.get_simulator()
            # sbibm simulators usually accept numpy arrays
            x_sim = simulator(valid_theta_phys) # [Valid_Batch, X_Dim]
            
            # 5. Compute Statistics & Distance
            # Convert to stats
            S_sim = torch.from_numpy(calculate_summary_stats(x_sim.numpy())).float()
            S_obs = x_obs_stats.cpu().view(1, -1) # [1, Stats_Dim]
            
            diff = S_sim - S_obs # [Valid_Batch, Stats_Dim]
            
            # Mahalanobis Distance: (x-u)T * Cov^-1 * (x-u)
            # Result: [Valid_Batch]
            cov_inv = stats_cov_inv.to(S_sim.device)
            dist_sq = torch.sum((diff @ cov_inv) * diff, dim=1)
            
            # 6. Combine: Score = Log_Prior - 0.5 * Distance
            # (Log_prior is typically 0 for uniform, but important for non-uniform)
            scale_factor = 2.0
            valid_scores = log_prior[valid_indices] - 0.5 * dist_sq * scale_factor
            
            # Assign back to main score tensor
            scores[valid_indices] = valid_scores
            
        except Exception as e:
            print(f"Simulator failed for some parameters: {e}")
            # Failed simulations remain -inf

    return scores.to(device)

def get_sbi_l2_scores(theta_pred_tensor, task, x_obs, standardize_fn, weights):
    """
    Score function: Log Prior - 0.5 * Weighted_Euclidean_Distance(Raw_Sim, Raw_Obs)
    Robust against NaNs and Simulator Failures.
    """
    batch_size = theta_pred_tensor.shape[0]
    device = theta_pred_tensor.device
    
    # 1. Un-standardize Theta
    theta_norm = theta_pred_tensor.detach().cpu()
    
    # Check for NaNs in Model Output
    if torch.isnan(theta_norm).any():
        # print("Warning: NaNs in model output theta.")
        return torch.full((batch_size,), -float('inf'), device=device)

    theta_phys = standardize_fn(theta_norm, label="theta", inverse=True)
    
    # 2. Prior Check
    try:
        log_prior = task.prior_dist.log_prob(theta_phys)
    except Exception as e:
        # Catch errors from prior (e.g. invalid input shape)
        return torch.full((batch_size,), -float('inf'), device=device)

    # Filter Valid Indices
    valid_indices = torch.isfinite(log_prior)
    valid_theta_phys = theta_phys[valid_indices]
    
    # Initialize scores with -inf
    scores = torch.full((batch_size,), -float('inf'), dtype=torch.float32).to(device)
    
    if valid_theta_phys.shape[0] > 0:
        # 3. Run Simulator
        simulator = task.get_simulator()
        # sbibm simulators usually accept numpy arrays
        x_sim = simulator(valid_theta_phys).to(device)
        
        # Ensure obs and weights shapes match
        # x_obs should be [1, Dim] or [Dim]
        x_obs_batch = torch.tensor(x_obs).to(device).view(1, -1)
        w_batch = torch.tensor(weights).to(device).view(1, -1)
        
        # 4. Weighted Euclidean Distance
        diff = x_sim - x_obs_batch
        
        # Calculate Distance Squared
        # Weighted Sum: sum( w_i * (sim_i - obs_i)^2 )
        dist_sq = torch.sum((diff ** 2) * w_batch, dim=1)
        
        # --- Robustness Check: Distance ---
        if torch.isnan(dist_sq).any():
            dist_sq = torch.nan_to_num(dist_sq, nan=float('inf'))
        
        # 5. Combine Score
        # Score = Log_Prior - 0.5 * Distance
        scale_factor = 1.0
        valid_scores = log_prior[valid_indices].to(device) - 0.5 * dist_sq * scale_factor
        
        scores[valid_indices] = valid_scores

    return scores.to(device)

def get_slcp_log_likelihood(theta, x_obs, task_name="slcp"):
    """
    计算 SLCP 任务的精确 Log Likelihood。
    不进行采样，直接评估分布密度。
    
    Args:
        theta: [Batch, 5] 物理参数
        x_obs: [1, 8] 或者是 [Batch, 8] 物理观测值 (由4个2D点展平)
    """
    batch_size = theta.shape[0]
    device = theta.device
    
    # 确保 x_obs 在正确的设备并扩展到 Batch
    # x_obs 通常是 [1, 8]，代表 4 个坐标点 (x1, y1, x2, y2, x3, y3, x4, y4)
    x_obs = torch.tensor(x_obs)
    if x_obs.dim() == 1:
        x_obs = x_obs.unsqueeze(0)
    
    # 如果 x_obs 是单条观测，扩展它以匹配 theta 的 batch
    if x_obs.shape[0] == 1 and batch_size > 1:
        x_obs = x_obs.expand(batch_size, -1)
        
    # --- 1. 复刻 Simulator 的参数变换逻辑 ---
    # SLCP parameters: theta has 5 dims
    # m (mean): theta[0], theta[1]
    # s1, s2 (scale): theta[2]^2, theta[3]^2
    # rho (correlation): tanh(theta[4])
    
    # 均值 m: [Batch, 2]
    m = theta[:, 0:2] 

    # 协方差参数
    s1 = theta[:, 2] ** 2
    s2 = theta[:, 3] ** 2
    rho = torch.tanh(theta[:, 4])

    # 构建协方差矩阵 S: [Batch, 2, 2]
    S = torch.zeros(batch_size, 2, 2, device=device)
    S[:, 0, 0] = s1 ** 2
    S[:, 0, 1] = rho * s1 * s2
    S[:, 1, 0] = rho * s1 * s2
    S[:, 1, 1] = s2 ** 2

    # 添加数值稳定性 (同 Simulator)
    eps = 1e-6
    S[:, 0, 0] += eps
    S[:, 1, 1] += eps

    # --- 2. 构建分布并计算 Log Prob ---
    # 观测数据包含 N=4 个点，每个点是 2D 的
    # x_obs shape [Batch, 8] -> Reshape to [Batch, 4, 2]
    num_points = 4
    x_obs_reshaped = x_obs.view(batch_size, num_points, 2).to(device)
    
    # 均值需要扩展到 [Batch, 1, 2] 以便广播到 4 个点
    # 协方差需要扩展到 [Batch, 1, 2, 2]
    dist = pdist.MultivariateNormal(loc=m.unsqueeze(1), covariance_matrix=S.unsqueeze(1))
    
    # 计算 log_prob
    # dist.log_prob(x) 会返回 [Batch, 4] (每个点的 log prob)
    log_probs_per_point = dist.log_prob(x_obs_reshaped)
    
    # --- 3. 求和得到总似然 ---
    # 假设 4 个观测点是独立同分布的 (i.i.d given theta)
    total_log_likelihood = torch.sum(log_probs_per_point, dim=1) # [Batch]
    
    return total_log_likelihood

def get_slcp_scores(theta_pred_tensor, task, x_obs, standardize_fn):
    """
    专门针对 SLCP 的 Score Function。
    Score = Log Prior + Exact Log Likelihood
    """
    batch_size = theta_pred_tensor.shape[0]
    device = theta_pred_tensor.device
    
    # 1. 反归一化得到物理参数
    theta_norm = theta_pred_tensor.detach().cpu()
    # 确保没有 NaN
    if torch.isnan(theta_norm).any():
        return torch.full((batch_size,), -float('inf'), device=device)
        
    theta_phys = standardize_fn(theta_norm, label="theta", inverse=True).to(device)
    
    # 2. 计算 Log Prior
    # 这里的 log_prob 是先验概率
    try:
        # 注意：sbibm 的 prior 可能会在 cpu 上计算，视 task 实现而定
        # 如果报错，可能需要转到 cpu: theta_phys.cpu()
        log_prior = task.prior_dist.log_prob(theta_phys.cpu()).to(device)
    except:
        return torch.full((batch_size,), -float('inf'), device=device)
    
    valid_indices = torch.isfinite(log_prior)
    
    scores = torch.full((batch_size,), -float('inf'), dtype=torch.float32, device=device)
    
    if valid_indices.any():
        # 3. 计算精确 Log Likelihood (核心改动)
        # 传入有效的物理参数和物理观测值
        # x_obs 应该是物理空间的 Tensor
        valid_theta = theta_phys[valid_indices]
        
        log_likelihood = get_slcp_log_likelihood(valid_theta, x_obs)
        
        # 4. Score = Log Prior + Log Likelihood
        scores[valid_indices] = log_prior[valid_indices] + log_likelihood
        
    return scores

def get_lv_log_likelihood(theta, x_obs, t_span=None, u0=None):
    """
    计算 Lotka-Volterra (subsample模式) 的 Log Likelihood。
    
    Args:
        theta: [Batch, 4] 参数 (alpha, beta, gamma, delta)
        x_obs: [Batch, 2, 10] 观测数据 (或者是 [Batch, 20] flatten后的)
    """
    batch_size = theta.shape[0]
    device = theta.device
    
    # 1. 初始条件和时间跨度 (与 sbibm 保持一致)
    if t_span is None:
        # days=20, saveat=0.1 -> 201 个点
        t_span = torch.linspace(0.0, 20.0, 201).to(device)
    
    if u0 is None:
        u0 = torch.tensor([30.0, 1.0]).to(device)
        u0 = u0.unsqueeze(0).expand(batch_size, -1)

    # 2. 定义 ODE (PyTorch版)
    class LotkaVolterraODE(torch.nn.Module):
        def __init__(self, theta):
            super().__init__()
            self.theta = theta

        def forward(self, t, u):
            x, y = u[:, 0], u[:, 1]
            alpha, beta, gamma, delta = self.theta[:, 0], self.theta[:, 1], self.theta[:, 2], self.theta[:, 3]
            dx = alpha * x - beta * x * y
            dy = -gamma * y + delta * x * y
            return torch.stack([dx, dy], dim=1)

    # 3. 求解 ODE
    try:
        # 求解器配置尽可能接近 sbibm 的 Julia 设置
        # 注意：这里我们只做前向计算，不涉及反向传播优化参数，所以不需要 adjoint
        with torch.no_grad():
            func = LotkaVolterraODE(theta)
            # sol: [Time=201, Batch, 2]
            sol = odeint(func, u0, t_span, method='rk4', rtol=1e-5, atol=1e-5)
            
            # [Time, Batch, 2] -> [Batch, 2, Time]
            u_sim = sol.permute(1, 2, 0)
            
            # --- 关键步骤：Subsample (复刻源码逻辑) ---
            # 源码: us[:, :, ::21]
            u_sub = u_sim[..., ::21] 
            
            # 数值稳定性处理 (同源码 clamp)
            # clamp(1e-10, 10000.0)
            u_sub = u_sub.clamp(1e-10, 10000.0)
            
    except Exception:
        return torch.full((batch_size,), -float('inf'), device=device)

    # 4. 计算 Log Likelihood (LogNormal Noise)
    x_obs = torch.tensor(x_obs).to(device)
    # 处理 x_obs 形状
    # 如果 x_obs 是单条 [1, 20] 或 [1, 2, 10]，广播匹配 batch
    if x_obs.shape[0] == 1 and batch_size > 1:
        if x_obs.dim() == 2: x_obs = x_obs.view(1, 2, 10)
        x_obs = x_obs.expand(batch_size, -1, -1)

    # 检查 x_obs 合法性 (LogNormal 要求 x > 0)
    if (x_obs <= 0).any():
        # 如果观测值有 <= 0，说明数据异常或者不是 LogNormal 适用的
        # 但 sbibm 生成的数据应该都是正的
        return torch.full((batch_size,), -float('inf'), device=device)

    # 定义分布: LogNormal(loc=log(u_sub), scale=0.1)
    # 注意：LogNormal 的 loc 参数是 log(mean)，不是 mean
    dist = pdist.LogNormal(loc=torch.log(u_sub), scale=0.1)
    
    # 计算 log_prob
    # log_probs: [Batch, 2, 10]
    log_probs = dist.log_prob(x_obs)
    
    # 求和 (假设独立)
    total_log_l = torch.sum(log_probs, dim=(1, 2))
    
    return total_log_l

def get_lv_scores(theta_pred_tensor, task, x_obs, standardize_fn):
    """
    Score = Log Prior + ODE-based Log Likelihood
    """
    batch_size = theta_pred_tensor.shape[0]
    device = theta_pred_tensor.device
    
    # 1. Un-standardize
    theta_norm = theta_pred_tensor.detach().cpu()
    theta_phys = standardize_fn(theta_norm, label="theta", inverse=True).to(device)
    
    # 2. Prior Check
    try:
        log_prior = task.prior_dist.log_prob(theta_phys.cpu()).to(device)
    except:
        return torch.full((batch_size,), -float('inf'), device=device)
        
    valid_indices = torch.isfinite(log_prior)
    scores = torch.full((batch_size,), -float('inf'), dtype=torch.float32, device=device)
    
    if valid_indices.any():
        valid_theta = theta_phys[valid_indices]
        
        # 3. 计算似然 (这里调用上面的 ODE 函数)
        # 注意：x_obs 需要是物理空间的原始时间序列
        log_likelihood = get_lv_log_likelihood(valid_theta, x_obs)
        
        # 4. Combine
        scores[valid_indices] = log_prior[valid_indices] + log_likelihood
        
    return scores

def _async_metric_worker(obs, post_n_cpu, ref_n_cpu, train_dir):
    """
    后台执行的 Worker，负责计算指标和画图。
    输入必须已经是 CPU 上的 Tensor/Array。
    """
    # 限制子进程里的 PyTorch 线程数，防止 CPU 争抢导致死机
    torch.set_num_threads(1) 
    
    results = {}
    
    # --- 1. Compute Metrics ---
    try:
        # 确保输入是 tensor (如果是 numpy 则转换，视你的 metric 函数要求而定)
        # 这里假设 c2st/mmd 接受 CPU tensor
        val_c2st = c2st(post_n_cpu, ref_n_cpu).item()
        val_mmd = mmd(post_n_cpu, ref_n_cpu).item()
        val_pme = posterior_mean_error(post_n_cpu, ref_n_cpu).item()
        
        results = {
            "obs": obs,
            "C2ST": val_c2st,
            "MMD": val_mmd,
            "PME": val_pme,
            "success": True
        }
        print(f"Obs {obs} [Background] | C2ST: {val_c2st:.4f}")
        
    except Exception as e:
        print(f"Obs {obs} [Background] | Metric calculation failed: {e}")
        results = {"obs": obs, "success": False}
        val_c2st = "N/A"

    # --- 2. Plotting ---
    try:
        fig = plt.figure(figsize=(10, 10))
        post_np = post_n_cpu.numpy()
        ref_np = ref_n_cpu.numpy()
        
        # Plot first 2 dimensions
        plt.scatter(post_np[:, 0], post_np[:, 1], s=2, alpha=0.5, label="Flow (MCMC)")
        plt.scatter(ref_np[:, 0], ref_np[:, 1], s=2, alpha=0.5, label="Reference")
        plt.legend()
        plt.title(f"Obs {obs} C2ST: {val_c2st}")
        plt.savefig(join(train_dir, f"posterior_{obs}.png"))
        plt.close(fig)
    except Exception as e:
        print(f"Obs {obs} [Background] | Plotting failed: {e}")

    return results

def evaluate_model_mcmc(train_dir, settings, dataset, model, use_wandb=False):
    task = sbibm.get_task(settings["task"]["name"])
    is_gpu = torch.cuda.is_available()
    device = torch.device("cuda" if is_gpu else "cpu")

    # --- Pre-computation ---
    # 1. Compute Inverse Covariance from Training Data for Mahalanobis Distance
    stats_cov_inv = get_covariance_from_dataset(dataset).to(device)

    # Initialize Metrics Containers
    metrics = {k: {} for k in ["C2ST", "MMD", "KSD", "PME", "PVR", "MEDDIST", "KL", "Sinkhorn"]}
    infer_time = {}
    all_score_histories = {} 

    for obs in range(1, 11):
        print(f"--- Evaluating Observation {obs} ---")
        
        # 1. Prepare Reference & Observation
        reference_samples = task.get_reference_posterior_samples(num_observation=obs).to(device)
        num_target_samples = len(reference_samples)
        
        # Get physical observation and compute its statistics
        x_obs_phys = task.get_observation(num_observation=obs).cpu().numpy()
        x_obs_stats = torch.from_numpy(calculate_summary_stats(x_obs_phys)).float().to(device)
        
        # Prepare standardized observation for Flow Context
        obs_tensor = torch.from_numpy(x_obs_phys).float().to(device)
        context_standardized = dataset.standardize(obs_tensor, label="x")
        print(f"{obs},context_standardized.shape:{context_standardized.shape}")
        # 2. Define Score Closure
        # def score_fn_closure(theta_pred_tensor):
        #     return get_sbi_scores(
        #         theta_pred_tensor, 
        #         task=task, 
        #         x_obs_stats=x_obs_stats,
        #         standardize_fn=dataset.standardize,
        #         stats_cov_inv=stats_cov_inv
        #     )
        def score_fn_closure(theta_pred_tensor):
            return get_lv_scores(
                theta_pred_tensor, 
                task=task, 
                x_obs=x_obs_phys,     # 传入原始物理观测值
                standardize_fn=dataset.standardize,
                # weights=get_data_weights_from_dataset(dataset).to(device)  # 传入计算好的权重
            )

        # 3. Run Sampling (MCMC/SMC Guided) - Batched
        start_time = time.time()
        
        # --- Batching Logic Start ---
        n_batches = 1
        batch_size = math.ceil(num_target_samples / n_batches)
        
        posterior_samples_list = []
        t1_avg_list = []
        t1_std_list = []
        print(f"Sampling {num_target_samples} samples in {n_batches} batches (batch size: {batch_size})...")

        # 异步任务列表
        futures = []
        # 创建进程池，max_workers 建议设为 CPU 核心数的一半或更少，留资源给 GPU 数据加载
        executor = concurrent.futures.ProcessPoolExecutor(max_workers=16)

        for i in range(n_batches):
            batch_samples, t1_avg_dict, t1_std_dict = model.sample(
                context_standardized, 
                num_samples=batch_size, 
                _score_fn=score_fn_closure
            )
            posterior_samples_list.append(batch_samples)
            t1_avg_list.append(t1_avg_dict)
            t1_std_list.append(t1_std_dict)
            # print(f"  Batch {i+1}/{n_batches} finished.")

        posterior_samples_norm = torch.cat(posterior_samples_list, dim=0)
        
        posterior_samples_norm = posterior_samples_norm[:num_target_samples]
        # --- Batching Logic End ---
        
        sampling_time = time.time() - start_time
        infer_time[f"infer_time_{obs}"] = sampling_time
        print(f"Sampling finished in {sampling_time:.2f}s. Total samples: {posterior_samples_norm.shape[0]}")

        # ==================================================================
        # 新增需求：合并 Score Dicts，画图并保存
        # ==================================================================
        
        # 1. 合并所有 Batch 的字典 (计算加权平均)
        # 假设每个 batch 的 step key 都是一样的
        if t1_avg_list and t1_std_list:
            aggregated_scores = {}
            aggregated_stds = {}
            
            # 获取所有出现的 step keys
            all_steps = list(t1_avg_list[0].keys())
            
            for step in all_steps:
                # 1. 收集所有 Batch 的 Mean 和 Std
                means = []
                stds = []
                for d_avg, d_std in zip(t1_avg_list, t1_std_list):
                    if step in d_avg and not math.isnan(d_avg[step]):
                        means.append(d_avg[step])
                        stds.append(d_std[step])
                
                if means:
                    # 计算全局平均的分数 (Mean of Means)
                    global_mean = sum(means) / len(means)
                    
                    # 计算全局标准差 (Pooled Variance 近似)
                    # 假设各 Batch 样本数相同，Pooled Std ≈ Sqrt(Sum(Std^2)/N) + Variance_between_means
                    # 这里为了简化展示，我们取各 Batch Std 的平均值作为"平均离散程度"
                    # 或者，如果你想展示所有粒子的总离散度，逻辑会更复杂
                    # 这里采用：平均标准差 (Average Spread)
                    global_std = sum(stds) / len(stds)
                    
                    aggregated_scores[step] = global_mean
                    aggregated_stds[step] = global_std
                else:
                    aggregated_scores[step] = float('nan')
                    aggregated_stds[step] = float('nan')
            
            # 2. 保存 CSV (包含 Mean 和 Std)
            score_data = []
            for step in aggregated_scores.keys():
                score_data.append({
                    'Step': step, 
                    'Mean_Score': aggregated_scores[step],
                    'Std_Score': aggregated_stds[step]
                })
                
            score_df = pd.DataFrame(score_data)
            
            # 排序
            score_df['Step_Numeric'] = pd.to_numeric(score_df['Step'], errors='coerce')
            score_df = score_df.sort_values(by='Step_Numeric')
            
            csv_path = join(train_dir, f"score_history_obs_{obs}.csv")
            score_df[['Step', 'Mean_Score', 'Std_Score']].to_csv(csv_path, index=False)
            
            # 3. 画图 (带误差带的折线图)
            fig_score = plt.figure(figsize=(12, 6))
            
            # 过滤掉 'final' 这种非数字 Step 用于画趋势
            plot_df = score_df.dropna(subset=['Step_Numeric'])
            x = plot_df['Step_Numeric']
            y = plot_df['Mean_Score']
            y_std = plot_df['Std_Score']
            
            # 绘制主曲线
            plt.plot(x, y, color='#1f77b4', marker='o', markersize=4, label='Mean Top-1 Score')
            
            # 绘制误差带 (Mean ± Std)
            # alpha 设置透明度，让图表更美观
            plt.fill_between(x, y - y_std, y + y_std, color='#1f77b4', alpha=0.2, label='Score Std Dev')
            
            # 标记 Final Score
            if 'final' in aggregated_scores:
                final_val = aggregated_scores['final']
                final_std = aggregated_stds['final']
                
                # 画一条虚线表示最终值
                plt.axhline(y=final_val, color='r', linestyle='--', linewidth=1.5, label=f'Final Mean: {final_val:.2f}')
                
                # 可选：在右侧标记最终分布范围
                plt.axhspan(final_val - final_std, final_val + final_std, color='r', alpha=0.1)
            
            plt.xlabel('Sampling Step', fontsize=12)
            plt.ylabel('Log Posterior Score', fontsize=12)
            plt.title(f'Score Evolution with Uncertainty (Obs {obs})', fontsize=14)
            plt.grid(True, alpha=0.3, linestyle='--')
            plt.legend(loc='lower right')
            
            # 优化布局
            plt.tight_layout()
            
            plt.savefig(join(train_dir, f"score_evolution_{obs}.png"), dpi=300)
            plt.close(fig_score)
            
            # 存入汇总
            all_score_histories[f'obs_{obs}'] = score_df

        # 4. Post-process Samples
        # Un-standardize to physical space
        posterior_samples = dataset.standardize(
            posterior_samples_norm, label="theta", inverse=True
        )

        # Final Prior Check (Sanity Check)
        prior_mask = torch.isfinite(task.prior_dist.log_prob(posterior_samples.cpu()))
        posterior_samples = posterior_samples[prior_mask].detach().to(device)
        
        # Match lengths for metrics
        n = min(len(reference_samples), len(posterior_samples))
        post_n = posterior_samples[:n]
        ref_n = reference_samples[:n]

        # A. 准备数据：截断并移动到 CPU
        # 必须在这里 .cpu()，否则 CUDA tensor 传不到子进程
        n = min(len(reference_samples), len(posterior_samples))
        
        # 使用 .detach().cpu() 彻底断开计算图并移至内存
        post_n_cpu = posterior_samples[:n].detach().cpu()
        ref_n_cpu = reference_samples[:n].detach().cpu()

        print(f"Submitting Obs {obs} metrics task to background...")
        
        # B. 提交任务给进程池
        # 注意：不要传递 dataset/task/model 这种大对象，只传必要的数据
        future = executor.submit(
            _async_metric_worker, 
            obs=obs, 
            post_n_cpu=post_n_cpu, 
            ref_n_cpu=ref_n_cpu, 
            train_dir=train_dir
        )
        futures.append(future)
        # 5. Compute Metrics
        # try:
        #     metrics["C2ST"][f"C2ST_{obs}"] = c2st(post_n, ref_n).item()
        #     metrics["MMD"][f"MMD_{obs}"]   = mmd(post_n, ref_n).item()
        #     metrics["PME"][f"PME_{obs}"]   = posterior_mean_error(post_n, ref_n).item()
            
        #     print(f"Obs {obs} | C2ST: {metrics['C2ST'][f'C2ST_{obs}']:.4f}")
        # except Exception as e:
        #     print(f"Metric calculation failed: {e}")

        # # 6. Plotting
        # fig = plt.figure(figsize=(10, 10))
        # post_np = post_n.cpu().numpy()
        # ref_np = ref_n.cpu().numpy()
        
        # # Plot first 2 dimensions
        # plt.scatter(post_np[:, 0], post_np[:, 1], s=2, alpha=0.5, label="Flow (MCMC)")
        # plt.scatter(ref_np[:, 0], ref_np[:, 1], s=2, alpha=0.5, label="Reference")
        # plt.legend()
        # plt.title(f"Obs {obs} C2ST: {metrics['C2ST'].get(f'C2ST_{obs}', 'N/A')}")
        # plt.savefig(join(train_dir, f"posterior_{obs}.png"))
        # plt.close(fig)

    print("\nAll GPU tasks finished. Waiting for background metric tasks...")
        
    # 等待所有任务完成并收集结果
    for future in concurrent.futures.as_completed(futures):
        try:
            res = future.result() # 这里会阻塞直到该任务完成
            if res["success"]:
                obs_idx = res["obs"]
                metrics["C2ST"][f"C2ST_{obs_idx}"] = res["C2ST"]
                metrics["MMD"][f"MMD_{obs_idx}"] = res["MMD"]
                metrics["PME"][f"PME_{obs_idx}"] = res["PME"]
        except Exception as e:
            print(f"A background task failed with error: {e}")
    executor.shutdown()
    # Save CSVs
    def save_csv(name, data):
        if not data: return
        with open(join(train_dir, name), "w") as f:
            w = csv.DictWriter(f, data.keys())
            w.writeheader()
            w.writerow(data)

    save_csv("c2st.csv", metrics["C2ST"])
    save_csv("mmd.csv", metrics["MMD"])
    save_csv("pme.csv", metrics["PME"])
    save_csv("inference_time.csv", infer_time)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_dir", required=True, help="Base save directory for the evaluation"
    )

    args = parser.parse_args()

    with open(join(args.train_dir, "settings.yaml"), "r") as f:
        settings = yaml.safe_load(f)

    use_wandb = settings["training"].get("wandb")
    if use_wandb:
        import wandb
        wandb.init(config=settings, dir=args.train_dir, **settings["training"]["wandb"])

    if settings["task"]["name"] == "lotka_volterra":
        dataset = load_dataset(directory_save="/home/vrlab/qinwch/flow_matching_record/experiment/lotka_volterra", settings=settings)
    else:
        dataset = generate_dataset(settings)

    train_loader, test_loader = build_train_and_test_loaders(
        dataset,
        settings["training"]["train_fraction"],
        settings["training"]["batch_size"],
        settings["training"]["num_workers"],
    )

    # model = train_model(
    #     args.train_dir,
    #     settings=settings,
    #     train_loader=train_loader,
    #     test_loader=test_loader,
    #     use_wandb=use_wandb,
    # )

    model = build_model_from_kwargs(
        # filename=join(args.train_dir, "best_model.pt"),
        filename=join(args.train_dir, "model_latest.pt"),
        device=settings["training"].get("device", "cpu"),
    )
    model.network.eval()
    # evaluate_model(args.train_dir, settings, dataset, model, use_wandb=use_wandb)
    evaluate_model_mcmc(args.train_dir, settings, dataset, model, use_wandb=use_wandb)
