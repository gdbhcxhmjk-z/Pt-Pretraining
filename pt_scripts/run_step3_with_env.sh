#!/bin/bash

# ================= 配置区 =================
# 原始数据根目录
RAW_DATA_ROOT="/bohr/pt-data-90l5/v3/"
TASK_NAME="pt_data"

# 【关键】组合出包含 lmdb 和 dict 的完整目录
DATA_PATH_FULL="${RAW_DATA_ROOT}/${TASK_NAME}"

# 模型路径
CKPT_PATH="${RAW_DATA_ROOT}/${TASK_NAME}/save_step1_structure/checkpoint_best.pt"

# 输出路径
SAVE_DIR="/share/model/pt_data/active_learning/round1"
N_CLUSTERS=200

# ================= Conda 环境初始化 =================
# 自动寻找 conda.sh 以便在脚本中使用 conda activate
if [ -f "/opt/mamba/etc/profile.d/conda.sh" ]; then
    source "/opt/mamba/etc/profile.d/conda.sh"
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    source "/opt/conda/etc/profile.d/conda.sh"
else
    echo "Warning: Could not find conda.sh."
fi

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4

echo "=== Step 3: Cold Start (Two-Stage Strategy) ==="
echo "Working Dir: $(pwd)"
echo "Data Path:   $DATA_PATH_FULL"

# ----------------------------------------------------
# 阶段 1: 运行 Unimol 推理 (使用 Base 环境)
# ----------------------------------------------------
# echo ">>> [Stage 1] Activating Base Environment..."
# conda activate base

# echo "[Stage 1] Running Inference..."
# # 注意：直接调用当前目录下的脚本
# python step3_part1_inference.py \
#     --data-path "$DATA_PATH_FULL" \
#     --ckpt-path "$CKPT_PATH" \
#     --save-dir "$SAVE_DIR" \
#     --dict-name "dict.txt" \
#     --batch-size 256

# if [ $? -ne 0 ]; then
#     echo "Inference failed! Exiting."
#     exit 1
# fi

# ----------------------------------------------------
# 阶段 2: 运行 Faiss 聚类 (使用 Faiss 环境)
# ----------------------------------------------------
echo ">>> [Stage 2] Activating Faiss Environment..."
conda activate faiss

echo "[Stage 2] Running Clustering..."
python step3_part2_clustering.py \
    --save-dir "$SAVE_DIR" \
    --n-clusters $N_CLUSTERS

if [ $? -eq 0 ]; then
    echo "=== ALL DONE ==="
    echo "Final Indices: $SAVE_DIR/cold_start_indices.txt"
else
    echo "Clustering failed!"
fi