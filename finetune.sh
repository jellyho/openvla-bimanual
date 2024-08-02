#!/bin/sh

torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path "openvla/openvla-7b" \
  --data_root_dir /home/jellyho/tensorflow_datasets \
  --dataset_name bm_sim \
  --run_root_dir /home/jellyho/openvla_run \
  --adapter_tmp_dir /home/jellyho/openvla_tmp \
  --lora_rank 32 \
  --batch_size 4 \
  --grad_accumulation_steps 4 \
  --learning_rate 5e-4 \
  --image_aug True \
  --wandb_project LG_OpenVLA \
  --wandb_entity jellyho_ \
  --save_steps 1000