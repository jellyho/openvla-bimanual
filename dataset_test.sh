#!/bin/sh

torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/dataset_test.py \
  --vla_path "openvla/openvla-7b" \
  --data_root_dir /home/jellyho/tensorflow_datasets \
  --dataset_name onearm_clean_joint_pos \
  --run_root_dir /home/jellyho/openvla_run \
  --adapter_tmp_dir /home/jellyho/openvla_tmp \
  --lora_rank 64 \
  --batch_size 2 \
  --grad_accumulation_steps 4 \
  --learning_rate 5e-4 \
  --image_aug True \
  --wandb_project OneArmClean_OpenVLA \
  --wandb_entity jellyho_ \
  --save_steps 1000
