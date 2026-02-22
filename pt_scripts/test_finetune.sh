data_path="../pt_data"  # LMDB 所在的根目录
save_dir="../pt_data/test10K_step2_finetune"  # Step 2 模型保存路径

# 假设您开了4卡机
n_gpu=1
MASTER_PORT=10095


# 【关键】这里必须指向 Step 1 (结构预训练) 跑出来的最佳权重
weight_path="/personal/pt_data/save_step1_structure/checkpoint_best.pt" 

# 这里的 task_name 主要用于寻找数据文件夹 (../pt_data/pt_pretrain/)
task_name="10K" 
dict_name="$task_name/dict.txt"

# 【修改点 1】指定我们注册的 Loss 名称
loss_func="multi_task_reg_loss"

# 【修改点 2】定义回归目标 (必须与 LMDB 中的 key 一致)
target_names="homo,lumo,gap,dipole"
norm_path="$data_path/$task_name/property_norms.pkl"

lr=1e-5
epoch=40
dropout=0.1 # 建议给一点 dropout 防止过拟合
warmup=0.06
only_polar=0
conf_size=1
seed=0

local_batch_size=32  # 单卡 Batch Size，取决于显存大小 (32G显存通常能跑32-64)
global_batch_size=256 # 目标总 Batch Size
update_freq=`expr $global_batch_size / $local_batch_size / $n_gpu`
if [ $update_freq -lt 1 ]; then update_freq=1; fi

# 【修改点 3】选择最佳模型的监控指标
# 我们的 reg_loss.py 会记录 valid_loss (总MSE), valid_homo_mae, valid_lumo_mae 等
# 建议使用 valid_loss (越小越好) 或 valid_homo_mae
metric="valid_loss"

export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1


# 检查 Step 1 权重是否存在
if [ ! -f "$weight_path" ]; then
    echo "Error: Step 1 checkpoint not found at $weight_path"
    exit 1
fi

python -m torch.distributed.run --nproc_per_node=$n_gpu --master_port=$MASTER_PORT $(which unicore-train) $data_path \
       --task-name $task_name \
       --user-dir ../unimol \
       --train-subset train --valid-subset valid \
       --conf-size $conf_size \
       --num-workers 1 --ddp-backend=c10d \
       --dict-name $dict_name \
       --task mol_finetune --loss $loss_func --arch unimol_base \
       --optimizer adam --adam-betas "(0.9, 0.99)" --adam-eps 1e-6 --clip-norm 1.0 \
       --lr-scheduler polynomial_decay --lr $lr --warmup-ratio $warmup --max-epoch $epoch --batch-size $local_batch_size --pooler-dropout $dropout \
       --update-freq $update_freq --seed $seed \
       --fp16 --fp16-init-scale 4 --fp16-scale-window 256 \
       --log-interval 100 --log-format simple \
       --validate-interval 1 \
       --finetune-from-model $weight_path \
       --best-checkpoint-metric $metric --patience 20 \
       --save-dir $save_dir --only-polar $only_polar \
       --regression-target-names $target_names \
       --property-norms-path $norm_path --keep-last-epochs 5 