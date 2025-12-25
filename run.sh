#!/bin/bash
set -euo pipefail  # 严格模式：出错立即退出、未定义变量报错、管道错误传递
# 避免脚本中中文乱码（可选，根据系统编码调整）
export LC_ALL=en_US.UTF-8

# ====================== 核心配置区（务必根据实际情况修改）======================
# 1. 项目工作区目录（脚本执行的根目录，存放run_sbibm.py的目录）
WORKSPACE="/home/vrlab/GW/flow_matching/flow-matching-posterior-estimation/sbi-benchmark/"

# 2. Conda环境配置
CONDA_ENV_NAME="fmpe"  # 替换为你的Conda环境名（比如base/flowmatch/venv1）
CONDA_MINIFORGE_PATH=""        # 非必须：若conda未加入系统PATH，填conda安装路径（如~/miniforge3/etc/profile.d/conda.sh）

# 3. 批量执行的命令列表（每行一个，按顺序执行）
COMMANDS=(
  # MLP
  # "python run_sbibm.py --train_dir /home/vrlab/GW/flowmatching_record/experiment/MLP/bernoulli_glm"
  # "python run_sbibm.py --train_dir /home/vrlab/GW/flowmatching_record/experiment/MLP/bernoulli_glm_raw"
  # "python run_sbibm.py --train_dir /home/vrlab/GW/flowmatching_record/experiment/MLP/gaussian_linear"
  # "python run_sbibm.py --train_dir /home/vrlab/GW/flowmatching_record/experiment/MLP/gaussian_linear_uniform"
  # "python run_sbibm.py --train_dir /home/vrlab/GW/flowmatching_record/experiment/MLP/gaussian_mixture"

  "python run_sbibm.py --train_dir /home/vrlab/GW/flowmatching_record/experiment/MLP/lotka_volterra"
  # "python run_sbibm.py --train_dir /home/vrlab/GW/flowmatching_record/experiment/MLP/sir"

  # "python run_sbibm.py --train_dir /home/vrlab/GW/flowmatching_record/experiment/MLP/slcp"

  # "python run_sbibm.py --train_dir /home/vrlab/GW/flowmatching_record/experiment/MLP/slcp_distractors"

  # "python run_sbibm.py --train_dir /home/vrlab/GW/flowmatching_record/experiment/MLP/two_moons"
  
  # MM-DiT
  # "python run_sbibm.py --train_dir /home/vrlab/GW/flowmatching_record/experiment/MM-DiT/bernoulli_glm"
  # "python run_sbibm.py --train_dir /home/vrlab/GW/flowmatching_record/experiment/MM-DiT/bernoulli_glm_raw"
  # "python run_sbibm.py --train_dir /home/vrlab/GW/flowmatching_record/experiment/MM-DiT/gaussian_linear"
  # "python run_sbibm.py --train_dir /home/vrlab/GW/flowmatching_record/experiment/MM-DiT/gaussian_linear_uniform"
  # "python run_sbibm.py --train_dir /home/vrlab/GW/flowmatching_record/experiment/MM-DiT/gaussian_mixture"

  "python run_sbibm.py --train_dir /home/vrlab/GW/flowmatching_record/experiment/MM-DiT/lotka_volterra"
  # "python run_sbibm.py --train_dir /home/vrlab/GW/flowmatching_record/experiment/MM-DiT/sir"

  # "python run_sbibm.py --train_dir /home/vrlab/GW/flowmatching_record/experiment/MM-DiT/slcp"

  # "python run_sbibm.py --train_dir /home/vrlab/GW/flowmatching_record/experiment/MM-DiT/slcp_distractors"

  # "python run_sbibm.py --train_dir /home/vrlab/GW/flowmatching_record/experiment/MM-DiT/two_moons"
)

# 4. 日志文件路径（记录执行过程，方便排查问题）
LOG_FILE="${WORKSPACE}/conda_run_commands.log"

# ====================== 工具函数（无需修改）======================
# 带时间戳的日志打印（同时输出到终端和日志文件）
log() {
  local timestamp=$(date +'%Y-%m-%d %H:%M:%S')
  local msg="[$timestamp] $1"
  echo -e "$msg" | tee -a "$LOG_FILE"
}

# 检查命令是否存在
check_command() {
  if ! command -v "$1" &> /dev/null; then
    log "错误：未找到命令 '$1'，请确认是否安装并加入系统PATH！"
    exit 1
  fi
}

# ====================== 核心执行逻辑 ======================
log "===== 开始执行批量命令脚本（Conda版）====="

# 步骤1：切换到工作区
log "\n【步骤1/4】切换工作区到：$WORKSPACE"
if [ ! -d "$WORKSPACE" ]; then
  log "错误：工作区目录 $WORKSPACE 不存在！"
  exit 1
fi
cd "$WORKSPACE" || { log "错误：切换工作区失败！"; exit 1; }
log "✅ 工作区切换成功（当前目录：$(pwd)）"

# 步骤2：初始化Conda环境（解决脚本中识别不到conda的问题）
log "\n【步骤2/4】初始化Conda环境"
# 若指定了conda安装路径，先加载conda配置
if [ -n "$CONDA_MINIFORGE_PATH" ] && [ -f "$CONDA_MINIFORGE_PATH" ]; then
  source "$CONDA_MINIFORGE_PATH"
  log "已加载自定义Conda路径：$CONDA_MINIFORGE_PATH"
fi
# 检查conda是否安装
check_command "conda"
# 初始化Conda的bash hook（关键：让脚本识别conda activate命令）
eval "$(conda shell.bash hook)"
log "✅ Conda初始化成功（当前Conda版本：$(conda --version)）"

# 步骤3：激活指定的Conda环境
log "\n【步骤3/4】激活Conda环境：$CONDA_ENV_NAME"
# 检查环境是否存在
if ! conda info --envs | grep -q "^$CONDA_ENV_NAME\s"; then
  log "错误：Conda环境 '$CONDA_ENV_NAME' 不存在！"
  log "提示：可用 'conda env list' 查看所有已创建的环境"
  exit 1
fi
# 激活环境
conda activate "$CONDA_ENV_NAME"
log "✅ Conda环境激活成功（当前环境：$CONDA_DEFAULT_ENV）"
log "   当前Python路径：$(which python) | Python版本：$(python --version)"

# 步骤4：批量执行命令
log "\n【步骤4/4】开始执行命令列表（共${#COMMANDS[@]}条）"
for idx in "${!COMMANDS[@]}"; do
  cmd_idx=$((idx + 1))
  cmd="${COMMANDS[$idx]}"
  
  log "\n----- 执行第 $cmd_idx/${#COMMANDS[@]} 条命令 -----"
  log "命令内容：$cmd"
  
  # 执行命令并捕获结果
  if eval "$cmd"; then
    log "✅ 第 $cmd_idx 条命令执行成功！"
  else
    cmd_exit_code=$?
    log "❌ 第 $cmd_idx 条命令执行失败！退出码：$cmd_exit_code"
    log "失败命令：$cmd"
    # 可选：注释下面的exit 1，改为continue，让失败后继续执行后续命令
    exit $cmd_exit_code
  fi
done

# 执行完成
log "\n===== 所有命令执行完成！====="
log "📝 完整日志文件：$LOG_FILE"
log "💡 脚本执行结束（Conda环境未退出，脚本终止后自动恢复系统默认环境）"

# 可选：主动退出Conda环境（非必须，脚本结束后自动退出）
# conda deactivate
