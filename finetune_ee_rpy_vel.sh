#!/bin/sh

while :; do
    RDZV_PORT=$((10000 + RANDOM % 20000))
    # Check if the port is available
    (echo >/dev/tcp/localhost/$RDZV_PORT) &>/dev/null || break
done

srun --job-name=OVLA-bench --gres=gpu:$1 torchrun --rdzv_id=$SLURM_JOB_ID --rdzv_backend=static --master_port=$RDZV_PORT --nnodes 1 --nproc-per-node $1 vla-scripts/finetune.py \
  --vla_path "openvla/openvla-7b" \
  --data_root_dir /home/shared/vla_benchmark_rlds \
  --dataset_name vla_benchmark \
  --run_root_dir /home/jellyho/openvla_run \
  --adapter_tmp_dir /home/jellyho/openvla_tmp \
  --lora_rank 64 \
  --batch_size 2 \
  --grad_accumulation_steps 8 \
  --learning_rate 5e-4 \
  --image_aug true \
  --wandb_project VLA_BENCHMARK_OPENVLA \
  --wandb_entity jellyho_ \
  --save_steps 1000 \
  --window_size 1 \
  --future_action_window_size 7 \
  --max_steps 100000