from helpers import Prior, train_mmdit  # 确保这里的 train 是上面修改过的新 train
from lampe.data import H5Dataset
from orbitize import DATADIR, read_input
import argparse
import os
os.environ["cUDA_VISIBLE_DEVICES"] = "1"
# 确保导入了 MMDiTFMPE 和 FMPELoss 类定义，如果它们在另一个文件里
# from mmdit_fmpe import MMDiTFMPE, FMPELoss

def main(size):
    # 读取数据
    trainset = H5Dataset(f'/mnt/npe-astrometry-betapic/datasets/betapic-train.h5', batch_size=2048, shuffle=True)  
    validset = H5Dataset(f'/mnt/npe-astrometry-betapic/datasets/betapic-val.h5', batch_size=2048, shuffle=True)

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
        epochs=1024,
        num_obs=num_obs,
        # --- MM-DiT 配置 ---
        hidden_dim=64,       # 根据显存大小调整，MMDiT 比较吃显存
        depth=4,             # Transformer 层数
        heads=4,             # 注意力头数
        context_num_tokens=8,# 将 context 向量切分为多少个 token
        theta_num_tokens=4,  # 将 theta 向量切分为多少个 token
        initial_lr=5e-4,     # Transformer 通常需要比 ResNet 更小的学习率
        clip=1.0             # 梯度裁剪对 Transformer 很重要
    )   

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train betapic MM-DiT model")
    parser.add_argument("--size", type=int, default=23, help="The exponent of 2 for the training dataset size")
    args = parser.parse_args()

    main(args.size)