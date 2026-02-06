import argparse
import yaml
from os.path import join
import wandb
from orbitize import DATADIR, read_input
import h5py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from dingo.core.posterior_models.build_model import (
    build_model_from_kwargs,
    autocomplete_model_kwargs,
)
from dingo.core.utils import build_train_and_test_loaders, RuntimeLimits
from helpers import Prior, train

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


class BetaPicDataset(Dataset):
    """
    H5 -> (theta_processed, x) 的 PyTorch Dataset

    注意：
    - 不接收 shuffle（shuffle 属于 DataLoader 参数）
    - 默认把数据放 CPU，避免 __getitem__ 内频繁 .to(device)
    """

    def __init__(
        self,
        h5_path: str,
        prior,
        theta_key: str = "theta",
        x_key: str = "x",
        dtype: torch.dtype = torch.float32,
        preprocess_in_init: bool = False,  # 大数据建议 False，小数据可 True 加速
    ):
        super().__init__()
        self.h5_path = h5_path
        self.prior = prior
        self.theta_key = theta_key
        self.x_key = x_key
        self.dtype = dtype
        self.preprocess_in_init = preprocess_in_init

        # 读入内存（你现在 batch_size=8192，通常数据不算太大，OK）
        with h5py.File(self.h5_path, "r") as f:
            theta_np = f[self.theta_key][:]
            x_np = f[self.x_key][:]

        assert len(theta_np) == len(x_np), "theta 和 x 的样本数不匹配！"

        self.theta = torch.from_numpy(theta_np).to(dtype=self.dtype)  # CPU
        self.x = torch.from_numpy(x_np).to(dtype=self.dtype)          # CPU

        # 可选：提前预处理所有 theta（若 prior.pre_process 是纯确定性且数据不大）
        if self.preprocess_in_init:
            with torch.no_grad():
                self.theta = self.prior.pre_process(self.theta)

    def __len__(self):
        return self.theta.shape[0]

    def __getitem__(self, idx: int):
        theta = self.theta[idx]
        x = self.x[idx]

        # 若未在 init 中预处理，这里对单样本预处理
        if not self.preprocess_in_init:
            theta = self.prior.pre_process(theta)

        return theta, x


priors = Prior()

trainset = BetaPicDataset(
    h5_path="/102437/qinwch/flow_matching/datasets/betapic-train.h5",
    prior=priors,
    preprocess_in_init=True,   # 小数据可 True
)

validset = BetaPicDataset(
    h5_path="/102437/qinwch/flow_matching/datasets/betapic-test.h5",
    prior=priors,
    preprocess_in_init=True,
)

train_loader = DataLoader(
    trainset,
    batch_size=4096,
    shuffle=True,          # ✅ 放这里
    num_workers=6,         # 视机器调整
    pin_memory=True,       # 若训练在 GPU 推荐 True
)

test_loader = DataLoader(
    validset,
    batch_size=4096,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)


num_obs = 2*(len(read_input.read_file('{}/betaPic.csv'.format(DATADIR))) - 1) # The RV observation is discarded

train_dir = "/102437/qinwch/flow_matching_record/orbitize_fmpe"
with open(join(train_dir, "settings.yaml"), "r") as f:
    settings = yaml.safe_load(f)

use_wandb = settings["training"].get("wandb")
if use_wandb:
    wandb.init(config=settings, dir=train_dir, **settings["training"]["wandb"])

model = train_model(
    train_dir,
    settings=settings,
    train_loader=train_loader,
    test_loader=test_loader,
    use_wandb=use_wandb,
)

# model = build_model_from_kwargs(
#     # filename=join(args.train_dir, "best_model.pt"),
#     filename=join(args.train_dir, "model_latest.pt"),
#     device=settings["training"].get("device", "cpu"),
# )
