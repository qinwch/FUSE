#!/bin/bash
set -euo pipefail
export LC_ALL=en_US.UTF-8

# ====================== 核心配置区 ======================
# 1. 路径配置
WORKSPACE="/home/qinwch/flow_matching/flow-matching-posterior-estimation/sbi-benchmark/"
# BASE_RECORD_DIR="/home/qinwch/flowmatching_record/experiment/20260113_fmpe_10_3"
# BASE_RECORD_DIR="/home/qinwch/flowmatching_record/experiment/20260113_fmpe_10_4"
BASE_RECORD_DIR="/home/qinwch/flowmatching_record/experiment/plot_all"

# 2. Conda配置
CONDA_ENV_NAME="fmpe"
CONDA_PROFILE="/home/vrlab/miniforge3/etc/profile.d/conda.sh" # 建议明确指向你的conda profile

# 3. 实验矩阵配置 (在这里修改模型和任务)
# MODELS=("MLP" "MM-DiT" "NPE")
# MODELS=("dims_32" "dims_64" "dims_128" "dims_256")
# MODELS=("lr_1e-4" "lr_2e-4" "lr_5e-4" "lr_1e-3" )
MODELS=("MM-DiT_10_4" "MM-DiT_10_3")

TASKS=(
  "bernoulli_glm"
  "bernoulli_glm_raw"
  "gaussian_linear"
  "gaussian_linear_uniform"
  "gaussian_mixture"
  "lotka_volterra"
  "sir"
  "slcp"
  "slcp_distractors"
  "two_moons"
)

# 4. 日志
LOG_FILE="${WORKSPACE}/run_fmpe.log"

# ====================== 工具函数 ======================
log() {
  echo -e "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# ====================== 执行逻辑 ======================
log "🚀 开始批量评估脚本"

# 步骤1：环境准备
cd "$WORKSPACE" || exit 1

if [ -f "$CONDA_PROFILE" ]; then
    source "$CONDA_PROFILE"
fi

eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV_NAME"
log "✅ 环境激活: $CONDA_DEFAULT_ENV (Python $(python -V 2>&1 | awk '{print $2}'))"

# 步骤2：计算任务总量
TOTAL_TASKS=$((${#MODELS[@]} * ${#TASKS[@]}))
CURRENT_COUNT=0

# 步骤3：嵌套循环执行矩阵
for model in "${MODELS[@]}"; do
    for task in "${TASKS[@]}"; do
        CURRENT_COUNT=$((CURRENT_COUNT + 1))
        
        # 构造训练目录路径
        TRAIN_DIR="${BASE_RECORD_DIR}/${model}/${task}"
        
        log "\n进度: [$CURRENT_COUNT/$TOTAL_TASKS] --------------------------"
        log "模型: $model | 任务: $task"
        log "路径: $TRAIN_DIR"

        # 检查目录是否存在
        if [ ! -d "$TRAIN_DIR" ]; then
            log "⚠️ 跳过：目录不存在 $TRAIN_DIR"
            continue
        fi

        # 执行命令
        log "▶️ 执行 run_sbibm.py..."
        if python run_sbibm.py --train_dir "$TRAIN_DIR"; then
            log "✅ 成功"
        else
            log "❌ 失败 (退出码: $?)"
            # 如果希望某个任务失败后停止整个脚本，取消下面注释
            # exit 1 
        fi
    done
done

log "\n🎉 所有任务处理完成！日志记录于: $LOG_FILE"