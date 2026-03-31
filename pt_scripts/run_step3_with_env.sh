#!/bin/bash

# ================= 配置区 =================
# 原始数据根目录
RAW_DATA_ROOT="/bohr/pt-data-90l5/v3/"
TASK_NAME="pt_data"

# 组合出包含 lmdb 和 dict 的完整目录
DATA_PATH_FULL="${RAW_DATA_ROOT}/${TASK_NAME}"

# 模型路径 (推理阶段已注释，此处保留变量以防未来需要)
CKPT_PATH="${RAW_DATA_ROOT}/${TASK_NAME}/save_step1_structure/checkpoint_best.pt"

# 输出路径 (注意：请确保此目录下已有之前生成的 step1_structure_embeddings.npy 和 ids.txt)
SAVE_DIR="/share/model/pt_data/active_learning/round1"
N_CLUSTERS=200
TOP_K=2  # 新增：每个聚类中心采样的分子数，200 * 2 = 400

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

echo "=== Step 3: Cold Start & Extraction ==="
echo "Working Dir: $(pwd)"
echo "Data Path:   $DATA_PATH_FULL"
echo "Save Dir:    $SAVE_DIR"

# ----------------------------------------------------
# 阶段 1: 运行 Unimol 推理 (已跳过)
# ----------------------------------------------------
# echo ">>> [Stage 1] Activating Base Environment..."
# conda activate base
# echo "[Stage 1] Running Inference..."
# python step3_part1_inference.py ...
# ... (保持注释即可) ...

echo ">>> [Stage 2] Activating Faiss Environment..."
conda activate faiss

# 生成 200 个聚类中心，每个保存 10 个备选
python step3_part2_clustering.py \
    --save-dir "$SAVE_DIR" \
    --n-clusters 200 \
    --buffer-k 10

if [ $? -ne 0 ]; then
    echo "Clustering failed! Exiting."
    exit 1
fi

# ----------------------------------------------------
# 阶段 3: 提取 XYZ 文件并进行电子数检查 (使用 Base 环境)
# ----------------------------------------------------
echo ">>> [Stage 3] Activating Base Environment (for LMDB)..."
conda activate base

XYZ_OUT_DIR="${SAVE_DIR}/molecules_xyz_filtered"

# 读取生成的 cold_start_candidates.npy，每个聚类强制选出合规的 2 个
python extract_xyz.py \
    --lmdb_path "${DATA_PATH_FULL}/train.lmdb" \
    --candidates_file "${SAVE_DIR}/cold_start_candidates.npy" \
    --out_dir "$XYZ_OUT_DIR" \
    --need_per_cluster 2