from helpers import Prior, train_mmdit  # 确保这里的 train 是上面修改过的新 train
from lampe.data import H5Dataset
from orbitize import DATADIR, read_input
import argparse
import os
from pathlib import Path
from path_config import DEFAULT_ORBIT_PATHS, OrbitPaths
# 确保导入了 MMDiTFMPE 和 FMPELoss 类定义，如果它们在另一个文件里
# from mmdit_fmpe import MMDiTFMPE, FMPELoss

def main(
    size,
    dataset_name,
    data_dir,
    models_dir,
    outputs_dir,
    epochs,
    batch_size,
    cuda_visible_devices,
):
    if cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

    paths = OrbitPaths(
        data_dir=Path(data_dir),
        models_dir=Path(models_dir),
        outputs_dir=Path(outputs_dir),
        references_dir=DEFAULT_ORBIT_PATHS.references_dir,
    )
    paths.ensure_directories()

    # 读取数据
    trainset = H5Dataset(str(paths.train_dataset(dataset_name)), batch_size=batch_size, shuffle=True)
    validset = H5Dataset(str(paths.val_dataset(dataset_name)), batch_size=batch_size, shuffle=True)

    priors = Prior() 

    # 计算观测数据维度
    # 注意：确保这里读取的文件和数据生成时的一致
    # BetaPic 通常有 PA 和 Sep 两个量，所以是 2 * (epochs - 1)
    num_obs = 2 * (len(read_input.read_file('{}/betaPic.csv'.format(DATADIR))) - 1) 

    print(f"Training MM-DiT on Dataset Size 2^{size} with Context Dim: {num_obs}")

    train_mmdit(
        trainset=trainset,
        validset=validset,
        prior=priors,
        epochs=epochs,
        num_obs=num_obs,
        # --- MM-DiT 配置 ---
        hidden_dim=64,       # 根据显存大小调整，MMDiT 比较吃显存
        depth=4,             # Transformer 层数
        heads=4,             # 注意力头数
        context_num_tokens=8,# 将 context 向量切分为多少个 token
        theta_num_tokens=4,  # 将 theta 向量切分为多少个 token
        initial_lr=5e-4,     # Transformer 通常需要比 ResNet 更小的学习率
        clip=1.0,            # 梯度裁剪对 Transformer 很重要
        save_path=str(paths.model_path(f"{dataset_name}_mmdit.pth")),
    )   

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train betapic MM-DiT model")
    parser.add_argument("--size", type=int, default=23, help="The exponent of 2 for the training dataset size")
    parser.add_argument("--dataset-name", type=str, default="orbit", help="Base name for datasets")
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_ORBIT_PATHS.data_dir), help="Directory containing HDF5 datasets")
    parser.add_argument("--models-dir", type=str, default=str(DEFAULT_ORBIT_PATHS.models_dir), help="Directory for trained model artifacts")
    parser.add_argument("--outputs-dir", type=str, default=str(DEFAULT_ORBIT_PATHS.outputs_dir), help="Directory for training outputs")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4096, help="Training and validation batch size")
    parser.add_argument("--cuda-visible-devices", type=str, default="", help="CUDA_VISIBLE_DEVICES value to set before training")
    args = parser.parse_args()

    main(
        args.size,
        args.dataset_name,
        args.data_dir,
        args.models_dir,
        args.outputs_dir,
        args.epochs,
        args.batch_size,
        args.cuda_visible_devices,
    )
