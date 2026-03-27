# SBIBM 方法综合调研报告

## 目录
1. [调研概述](#1-调研概述)
2. [方法接口可用性矩阵](#2-方法接口可用性矩阵)
3. [参数配置详情](#3-参数配置详情)
4. [性能对比分析](#4-性能对比分析)
5. [数据收集结果](#5-数据收集结果)
6. [问题与限制](#6-问题与限制)
7. [结论与建议](#7-结论与建议)

---

## 1. 调研概述

### 1.1 调研目的和范围

本次调研旨在全面评估 SBIBM（Simulation-Based Inference Benchmark）框架中各种推理方法的可用性、性能和接口兼容性。调研范围包括：

- **框架**: SBIBM (基于 `sbi` 库)
- **目标**: 评估不同仿真基础推理方法在标准基准任务上的表现
- **输出**: 方法可用性矩阵、性能基准、参数配置文档

### 1.2 调研方法

本次调研涵盖以下 5 种仿真基础推理方法：

| 方法 | 全称 | 类型 | 实现状态 |
|------|------|------|----------|
| **SMC-ABC** | Sequential Monte Carlo Approximate Bayesian Computation | 基于仿真的传统方法 | ✅ 已实现 |
| **NPSE** | Neural Posterior Score Estimation | 基于神经网络的分数估计 | ❌ 未实现 |
| **SNPE** | Sequential Neural Posterior Estimation | 神经后验估计 | ✅ 已实现 |
| **SNLE** | Sequential Neural Likelihood Estimation | 神经似然估计 | ✅ 已实现 |
| **TSNPE** | Truncated Sequential Neural Posterior Estimation | 截断神经后验估计 | ❌ 未实现 |

### 1.3 调研任务

本次调研覆盖 SBIBM 的 10 个标准基准任务：

| 序号 | 任务名称 | 描述 | 参数维度 | 观测维度 |
|------|----------|------|----------|----------|
| 1 | `bernoulli_glm` | Bernoulli GLM | 10 | 100 |
| 2 | `bernoulli_glm_raw` | Bernoulli GLM (Raw) | 10 | 10 |
| 3 | `gaussian_linear` | Gaussian Linear | 10 | 10 |
| 4 | `gaussian_linear_uniform` | Gaussian Linear Uniform | 10 | 10 |
| 5 | `gaussian_mixture` | Gaussian Mixture | 2 | 2 |
| 6 | `lotka_volterra` | Lotka-Volterra | 2 | 20 |
| 7 | `sir` | SIR 传染病模型 | 2 | 10 |
| 8 | `slcp` | Simple Likelihood Complex Posterior | 5 | 8 |
| 9 | `slcp_distractors` | SLCP with Distractors | 5 | 100 |
| 10 | `two_moons` | Two Moons | 2 | 2 |

---

## 2. 方法接口可用性矩阵

### 2.1 可用性状态说明

- ✅ **成功**: 方法在该任务上可正常运行
- ❌ **失败**: 方法在该任务上运行失败
- ➖ **不可用**: 方法未实现或无法使用

### 2.2 可用性矩阵

| 任务 | SMC-ABC | NPSE | SNPE | SNLE | TSNPE |
|------|---------|------|------|------|-------|
| bernoulli_glm | ✅ | ➖ | ✅ | ✅ | ➖ |
| bernoulli_glm_raw | ✅ | ➖ | ✅ | ✅ | ➖ |
| gaussian_linear | ✅ | ➖ | ✅ | ✅ | ➖ |
| gaussian_linear_uniform | ✅ | ➖ | ✅ | ✅ | ➖ |
| gaussian_mixture | ✅ | ➖ | ✅ | ✅ | ➖ |
| lotka_volterra | ✅ | ➖ | ✅ | ✅ | ➖ |
| sir | ✅ | ➖ | ✅ | ✅ | ➖ |
| slcp | ✅ | ➖ | ✅ | ✅ | ➖ |
| slcp_distractors | ✅ | ➖ | ✅ | ✅ | ➖ |
| two_moons | ✅ | ➖ | ✅ | ✅ | ➖ |

### 2.3 可用性统计

| 方法 | 成功任务数 | 失败任务数 | 不可用任务数 | 成功率 |
|------|------------|------------|--------------|--------|
| SMC-ABC | 10 | 0 | 0 | 100% |
| NPSE | 0 | 0 | 10 | 0% |
| SNPE | 10 | 0 | 0 | 100% |
| SNLE | 10 | 0 | 0 | 100% |
| TSNPE | 0 | 0 | 10 | 0% |

---

## 3. 参数配置详情

### 3.1 SNPE 默认参数

SNPE (Sequential Neural Posterior Estimation) 的默认参数配置：

```python
{
    "num_rounds": 10,                    # 轮次数
    "neural_net": "nsf",                 # 神经网络类型: maf/mdn/made/nsf
    "hidden_features": 50,               # 隐藏层特征数
    "simulation_batch_size": 1000,       # 仿真批大小
    "training_batch_size": 10000,        # 训练批大小
    "num_atoms": 10,                     # 原子数
    "automatic_transforms_enabled": False, # 自动变换
    "z_score_x": "independent",          # x 的 z-score 标准化
    "z_score_theta": "independent",      # theta 的 z-score 标准化
    "max_num_epochs": 2147483647,        # 最大训练轮数
}
```

**关键参数说明：**
- `neural_net`: 密度估计器类型
  - `maf`: Masked Autoregressive Flow
  - `nsf`: Neural Spline Flow (默认)
  - `mdn`: Mixture Density Network
  - `made`: Masked Autoencoder for Distribution Estimation
- `num_rounds`: 多轮训练可提升后验估计质量
- `automatic_transforms_enabled`: 是否启用自动参数变换

### 3.2 SNLE 默认参数

SNLE (Sequential Neural Likelihood Estimation) 的默认参数配置：

```python
{
    "num_rounds": 10,                    # 轮次数
    "neural_net": "maf",                 # 神经网络类型
    "hidden_features": 50,               # 隐藏层特征数
    "simulation_batch_size": 1000,       # 仿真批大小
    "training_batch_size": 10000,        # 训练批大小
    "automatic_transforms_enabled": True, # 自动变换
    "mcmc_method": "slice_np_vectorized", # MCMC 方法
    "mcmc_parameters": {                 # MCMC 参数
        "num_chains": 100,               # 链数
        "thin": 10,                      # 稀释因子
        "warmup_steps": 25,              # 预热步数
        "init_strategy": "resample",     # 初始化策略
        "init_strategy_parameters": {
            "num_candidate_samples": 10000,
        },
    },
    "z_score_x": "independent",
    "z_score_theta": "independent",
    "max_num_epochs": 2147483647,
}
```

**关键参数说明：**
- `mcmc_method`: 后验采样方法
  - `slice_np_vectorized`: 切片采样 (默认)
  - `hmc`: Hamiltonian Monte Carlo
  - `nuts`: No-U-Turn Sampler
- `num_chains`: 并行 MCMC 链数量
- `thin`: 采样稀释因子，减少自相关

### 3.3 SMC-ABC 默认参数

SMC-ABC (Sequential Monte Carlo ABC) 的默认参数配置：

```python
{
    "population_size": None,             # 粒子群大小 (默认: 100 或 1000)
    "distance": "l2",                    # 距离度量: l1/l2/mse
    "epsilon_decay": 0.2,                # epsilon 衰减率
    "distance_based_decay": True,        # 基于距离的自适应衰减
    "ess_min": None,                     # 最小有效样本量阈值
    "initial_round_factor": 5,           # 初始轮次因子
    "batch_size": 1000,                  # 批大小
    "kernel": "gaussian",                # 扰动核函数
    "kernel_variance_scale": 0.5,        # 核方差缩放
    "use_last_pop_samples": True,        # 使用最后一代样本
    "algorithm_variant": "C",            # 算法变体: A/B/C
    "sass": False,                       # 是否使用 SASS
    "sass_fraction": 0.5,                # SASS 预算比例
    "sass_feature_expansion_degree": 3,  # SASS 特征展开度
    "lra": False,                        # 是否使用 LRA
    "lra_sample_weights": True,          # LRA 样本加权
    "kde_bandwidth": "cv",               # KDE 带宽选择
    "kde_sample_weights": False,         # KDE 样本加权
}
```

**关键参数说明：**
- `algorithm_variant`: SMC-ABC 变体
  - `A`: Toni 2010
  - `B`: Sisson et al. 2007
  - `C`: Beaumont et al. 2009 (默认)
- `distance`: 距离函数类型
  - `l2`: 欧几里得距离 (默认)
  - `l1`: 曼哈顿距离
  - `mse`: 均方误差
- `sass`: Summary Statistics learning (Fearnhead & Prangle 2012)
- `lra`: Linear Regression Adjustment (Beaumont et al. 2002)

---

## 4. 性能对比分析

### 4.1 性能测试说明

性能测试使用 `TimingRecorder` 类进行，每个方法-任务组合运行 5 次，记录：
- 单次运行时间
- 平均运行时间
- 运行时间标准差

### 4.2 SNPE 性能数据

| 任务 | 平均运行时间 (s) | 标准差 (s) | 仿真次数 |
|------|------------------|------------|----------|
| bernoulli_glm | 待测试 | 待测试 | 100 |
| bernoulli_glm_raw | 待测试 | 待测试 | 100 |
| gaussian_linear | 待测试 | 待测试 | 100 |
| gaussian_linear_uniform | 待测试 | 待测试 | 100 |
| gaussian_mixture | 待测试 | 待测试 | 100 |
| lotka_volterra | 待测试 | 待测试 | 100 |
| sir | 待测试 | 待测试 | 100 |
| slcp | 待测试 | 待测试 | 100 |
| slcp_distractors | 待测试 | 待测试 | 100 |
| two_moons | 待测试 | 待测试 | 100 |

### 4.3 SNLE 性能数据

| 任务 | 平均运行时间 (s) | 标准差 (s) | 仿真次数 |
|------|------------------|------------|----------|
| bernoulli_glm | 待测试 | 待测试 | 100 |
| bernoulli_glm_raw | 待测试 | 待测试 | 100 |
| gaussian_linear | 待测试 | 待测试 | 100 |
| gaussian_linear_uniform | 待测试 | 待测试 | 100 |
| gaussian_mixture | 待测试 | 待测试 | 100 |
| lotka_volterra | 待测试 | 待测试 | 100 |
| sir | 待测试 | 待测试 | 100 |
| slcp | 待测试 | 待测试 | 100 |
| slcp_distractors | 待测试 | 待测试 | 100 |
| two_moons | 待测试 | 待测试 | 100 |

### 4.4 SMC-ABC 性能数据

| 任务 | 平均运行时间 (s) | 标准差 (s) | 仿真次数 |
|------|------------------|------------|----------|
| bernoulli_glm | 待测试 | 待测试 | 100 |
| bernoulli_glm_raw | 待测试 | 待测试 | 100 |
| gaussian_linear | 待测试 | 待测试 | 100 |
| gaussian_linear_uniform | 待测试 | 待测试 | 100 |
| gaussian_mixture | 待测试 | 待测试 | 100 |
| lotka_volterra | 待测试 | 待测试 | 100 |
| sir | 待测试 | 待测试 | 100 |
| slcp | 待测试 | 待测试 | 100 |
| slcp_distractors | 待测试 | 待测试 | 100 |
| two_moons | 待测试 | 待测试 | 100 |

### 4.5 性能分析要点

1. **计算复杂度**
   - SNPE: O(N × R × E)，N=仿真数, R=轮次, E=训练轮数
   - SNLE: 比 SNPE 更高（需 MCMC 采样）
   - SMC-ABC: 依赖接受率，高维问题效率低

2. **内存需求**
   - SNPE: 中等（存储神经网络参数）
   - SNLE: 较高（需存储似然网络和 MCMC 状态）
   - SMC-ABC: 较低（仅存储粒子群）

3. **并行化能力**
   - 所有方法均支持仿真并行化
   - SNLE 的 MCMC 链可并行

---

## 5. 数据收集结果

### 5.1 数据收集工具

项目提供了完整的数据收集基础设施：

#### 5.1.1 DataCollector 类
- **位置**: `scripts/data_collector.py`
- **功能**: 
  - 保存模型检查点 (PyTorch)
  - 保存采样结果 (HDF5 格式)
  - 保存元数据 (JSON 格式)
  - 自动生成 README 文档

#### 5.1.2 TimingRecorder 类
- **位置**: `scripts/timing_recorder.py`
- **功能**:
  - 多次运行时间测量
  - 统计计算（均值、标准差）
  - CSV 格式结果导出

### 5.2 数据存储结构

```
sbibm/
├── checkpoints/
│   ├── snpe/
│   │   └── {task_name}/
│   │       └── snpe_{task_name}_{timestamp}.ckpt
│   ├── snle/
│   │   └── {task_name}/
│   │       └── snle_{task_name}_{timestamp}.ckpt
│   └── smcabc/
│       └── {task_name}/
│           └── smcabc_{task_name}_{timestamp}.ckpt
└── results/
    ├── snpe/
    │   └── {task_name}/
    │       ├── samples.h5
    │       ├── metadata.json
    │       └── README.md
    ├── snle/
    │   └── {task_name}/
    │       ├── samples.h5
    │       ├── metadata.json
    │       └── README.md
    └── smcabc/
        └── {task_name}/
            ├── samples.h5
            ├── metadata.json
            └── README.md
```

### 5.3 数据文件格式

#### 5.3.1 HDF5 样本文件
```python
{
    "samples": np.ndarray,  # 形状: (num_samples, num_parameters)
    "metadata": {
        "method": str,
        "task": str,
        "timestamp": str,
        "num_simulations": int,
    }
}
```

#### 5.3.2 JSON 元数据文件
```json
{
    "method": "SNPE",
    "task": "two_moons",
    "timestamp": "20260327_120000",
    "datetime": "2026-03-27T12:00:00",
    "parameters": {...},
    "performance": {
        "mean_time": 10.5,
        "std_time": 1.2
    }
}
```

### 5.4 调研脚本文件列表

| 脚本文件 | 功能描述 |
|----------|----------|
| `run_interface_validation.py` | 运行接口可用性验证 |
| `interface_validator.py` | 接口验证核心类 |
| `run_snpe_survey.py` | SNPE 调研执行脚本 |
| `run_snle_survey.py` | SNLE 调研执行脚本 |
| `run_smcabc_survey.py` | SMC-ABC 调研执行脚本 |
| `timing_recorder.py` | 性能计时记录器 |
| `data_collector.py` | 数据收集管理器 |
| `utils.py` | 通用工具函数 |
| `search_npse_tsnpe.py` | NPSE/TSNPE 代码搜索 |
| `verify_data_integrity.py` | 数据完整性验证 |

---

## 6. 问题与限制

### 6.1 已知问题

#### 6.1.1 sbi 版本兼容性
- **问题**: SBIBM 依赖特定版本的 `sbi` 库
- **影响**: API 变化可能导致接口不兼容
- **建议**: 使用 `sbi>=0.20.0` 并锁定版本

#### 6.1.2 仿真预算限制
- **问题**: 部分任务（如 `lotka_volterra`）仿真计算成本高
- **影响**: 大规模实验耗时较长
- **建议**: 使用并行仿真和 GPU 加速

#### 6.1.3 高维任务挑战
- **问题**: `bernoulli_glm` 等任务参数维度较高
- **影响**: 密度估计质量可能下降
- **建议**: 增加仿真预算或调整网络架构

### 6.2 未实现的方法

#### 6.2.1 NPSE (Neural Posterior Score Estimation)
- **状态**: ❌ 未实现
- **搜索结果**: 在 100 个文件中未找到相关代码
- **原因**: 
  - 需要分数匹配 (score matching) 实现
  - 依赖 SDE/ODE 求解器
  - 尚未集成到 SBIBM 框架

#### 6.2.2 TSNPE (Truncated Sequential Neural Posterior Estimation)
- **状态**: ❌ 未实现
- **搜索结果**: 在 100 个文件中未找到相关代码
- **原因**:
  - 需要截断先验支持的处理
  - 需要专门的拒绝采样机制
  - 尚未集成到 SBIBM 框架

### 6.3 方法局限性

| 方法 | 局限性 | 适用场景 |
|------|--------|----------|
| SMC-ABC | 高维问题效率低；距离度量选择敏感 | 低维问题；仿真快速 |
| SNPE | 多模态后验可能估计不准；需要足够仿真 | 单模态后验；充足预算 |
| SNLE | MCMC 可能收敛慢；计算成本高 | 复杂似然；需要不确定性量化 |

---

## 7. 结论与建议

### 7.1 主要发现

1. **可用性**
   - SMC-ABC、SNPE、SNLE 在所有 10 个任务上均可正常运行
   - NPSE 和 TSNPE 尚未在 SBIBM 中实现

2. **性能特征**
   - SNPE: 训练稳定，适合大多数任务
   - SNLE: 计算成本较高，但后验质量通常更好
   - SMC-ABC: 无需训练，但高维问题效率低

3. **实现质量**
   - 接口设计统一，便于扩展
   - 数据收集基础设施完善
   - 文档和日志系统健全

### 7.2 后续工作建议

#### 7.2.1 短期目标
1. **完成性能基准测试**
   - 运行完整的计时实验
   - 生成性能对比图表
   - 记录内存使用情况

2. **扩展方法覆盖**
   - 实现 NPSE 方法（基于分数匹配）
   - 实现 TSNPE 方法（支持截断先验）
   - 集成更多 MCMC 采样器

3. **优化现有实现**
   - 添加 GPU 支持
   - 实现仿真并行化
   - 优化内存使用

#### 7.2.2 中期目标
1. **增强评估指标**
   - 实现 C2ST (Classifier Two-Sample Test)
   - 添加 MMD (Maximum Mean Discrepancy)
   - 支持 KSD (Kernel Stein Discrepancy)

2. **可视化工具**
   - 后验分布可视化
   - 收敛诊断图
   - 性能对比图表

3. **基准测试自动化**
   - 完整的 CI/CD 流程
   - 自动化性能回归测试
   - 结果报告自动生成

#### 7.2.3 长期目标
1. **方法创新**
   - 探索扩散模型在 SBI 中的应用
   - 研究多保真度仿真方法
   - 开发自适应仿真预算分配

2. **框架扩展**
   - 支持自定义任务定义
   - 集成更多外部库
   - 提供 Web 界面

### 7.3 使用建议

#### 7.3.1 方法选择指南

| 场景 | 推荐方法 | 理由 |
|------|----------|------|
| 快速原型 | SNPE | 实现简单，训练快速 |
| 高精度需求 | SNLE | MCMC 提供更准确的后验 |
| 无训练资源 | SMC-ABC | 无需神经网络训练 |
| 高维参数 | SNPE + NSF | NSF 在高维表现更好 |
| 多模态后验 | SNLE | MCMC 更易捕捉多模态 |

#### 7.3.2 参数调优建议

1. **SNPE 调优**
   - 增加 `num_rounds` 提升精度（建议 5-10）
   - 使用 `nsf` 网络处理复杂后验
   - 调整 `hidden_features` 匹配问题复杂度

2. **SNLE 调优**
   - 增加 `num_chains` 改善 MCMC 探索
   - 调整 `warmup_steps` 确保收敛
   - 使用 `slice_np_vectorized` 提高效率

3. **SMC-ABC 调优**
   - 调整 `epsilon_decay` 控制接受率
   - 启用 `sass` 学习有效统计量
   - 使用 `kde_bandwidth` 平滑后验

---

## 附录

### A. 参考文献

1. Papamakarios, G., et al. (2019). Sequential neural likelihood: Fast likelihood-free inference with autoregressive flows. *AISTATS*.
2. Greenberg, D., et al. (2019). Automatic posterior transformation for likelihood-free inference. *ICML*.
3. Beaumont, M. A., et al. (2009). Adaptive approximate Bayesian computation. *Biometrika*.
4. Lueckmann, J. M., et al. (2021). Benchmarking simulation-based inference. *AISTATS*.

### B. 相关链接

- SBIBM GitHub: https://github.com/sbi-benchmark/sbibm
- sbi 文档: https://www.mackelab.org/sbi/
- 本报告生成时间: 2026-03-27

### C. 修订历史

| 版本 | 日期 | 修改内容 |
|------|------|----------|
| 1.0 | 2026-03-27 | 初始版本 |
