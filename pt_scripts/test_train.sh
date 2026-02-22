data_path="../pt_data/10K" # replace to your data path
save_dir="../pt_data/test10K_step1_structure" # replace to your save path
n_gpu=1
MASTER_PORT=10091
lr=1e-4 
wd=1e-4 
batch_size=64 
update_freq=4
masked_token_loss=1
masked_coord_loss=5
masked_dist_loss=10
x_norm_loss=0.01
delta_pair_repr_norm_loss=0.01
mask_prob=0.15
only_polar=0
noise_type="uniform"
noise=1.0
seed=1
layers=15 #
warmup_steps=100
max_steps=1000

export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1
# 检查 dict.txt 是否存在，防止跑起来才报错
if [ ! -f "$data_path/dict.txt" ]; then
    echo "Error: dict.txt not found in $data_path"
    exit 1
fi

python -m torch.distributed.run --nproc_per_node=$n_gpu --master_port=$MASTER_PORT $(which unicore-train) $data_path  --user-dir ../unimol --train-subset train --valid-subset valid \
       --num-workers 2 --ddp-backend=c10d \
       --task unimol --loss unimol --arch unimol_base  \
       --optimizer adam --adam-betas "(0.9, 0.99)" --adam-eps 1e-6 --clip-norm 1.0 --weight-decay $wd \
       --lr-scheduler polynomial_decay --lr $lr --warmup-updates $warmup_steps --total-num-update $max_steps \
       --update-freq $update_freq --seed $seed \
       --fp16 --fp16-init-scale 4 --fp16-scale-window 256 --tensorboard-logdir $save_dir/tsb \
       --max-update $max_steps --log-interval 10 --log-format simple \
       --save-interval-updates 10000 --validate-interval-updates 10000 --keep-interval-updates 10 --no-epoch-checkpoints  \
       --masked-token-loss $masked_token_loss --masked-coord-loss $masked_coord_loss --masked-dist-loss $masked_dist_loss \
       --x-norm-loss $x_norm_loss --delta-pair-repr-norm-loss $delta_pair_repr_norm_loss \
       --mask-prob $mask_prob --noise-type $noise_type --noise $noise --batch-size $batch_size \
       --save-dir $save_dir  --only-polar $only_polar --encoder-layers $layers 