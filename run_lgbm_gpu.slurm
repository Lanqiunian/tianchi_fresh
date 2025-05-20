#!/bin/bash

#SBATCH -o job.%j.out
#SBATCH -e job.%j.err
#SBATCH -J tianchi_lgbm_gpu
#SBATCH --partition=gpu         # 关键：指定gpu分区
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8       # 根据需要调整
#SBATCH --mem=64G               # 根据需要调整 (GPU节点内存较多，有1024G)
#SBATCH -G 1                    # 关键：请求1块GPU (节点有4块A800)
#SBATCH --time=02:00:00         # 预计运行时间

echo "作业开始时间: $(date)"
echo "作业运行节点: $(hostname)"
echo "申请的GPU: $CUDA_VISIBLE_DEVICES" # Slurm通常会自动设置这个变量

# --- 环境设置 ---
echo "正在加载模块..."
source /home-ssd/hpc/ini_module.bash
module load env/anaconda3/2022.05  # 加载你选择的Anaconda环境
# --- 激活你的Conda环境 ---
# 确定conda的初始化脚本路径
# 通常是下面这样，但请根据您的Anaconda安装确认
CONDA_BASE_PATH=$(conda info --base)
if [ -f "$CONDA_BASE_PATH/etc/profile.d/conda.sh" ]; then
    source "$CONDA_BASE_PATH/etc/profile.d/conda.sh"
else
    echo "错误: 找不到 conda.sh 初始化脚本。请检查路径。"
    exit 1
fi

# --- 激活你的Conda环境 ---
# 如果 tianchi_project_env 是一个conda环境
# 请确保知道conda的初始化脚本路径和你的环境名称/路径
# 例如:
# CONDA_BASE_PATH=$(conda info --base) # 获取conda基础路径
# source $CONDA_BASE_PATH/etc/profile.d/conda.sh
# conda activate tianchi_project_env # 或者conda activate /path/to/your/env

echo "当前Python路径: $(which python)"
echo "当前conda环境: $CONDA_DEFAULT_ENV"

# --- 进入项目目录 ---
PROJECT_DIR="/home-ssd/Users/nsgm_liyx/tianchi" # 你的项目路径
cd $PROJECT_DIR
echo "当前工作目录: $(pwd)"

# --- 执行Python脚本 ---
echo "开始运行LightGBM训练脚本..."
python -m src.rank.lgbm_trainer
EXIT_CODE=$? # 获取Python脚本的退出码

if [ $EXIT_CODE -ne 0 ]; then
    echo "Python脚本执行失败，退出码: $EXIT_CODE"
    # 你可以在这里添加一些清理操作，如果需要的话
    exit $EXIT_CODE # 以相同的错误码退出Slurm作业
fi

echo "脚本执行完毕: $(date)"