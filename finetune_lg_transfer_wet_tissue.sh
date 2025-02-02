#!/bin/sh

torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path "openvla/openvla-7b" \
  --data_root_dir /home/jellyho/tensorflow_datasets \
  --dataset_name lg_transfer_wet_tissue \
  --run_root_dir /home/jellyho/openvla_run \
  --adapter_tmp_dir /home/jellyho/openvla_tmp \
  --lora_rank 64 \
  --batch_size 2 \
  --grad_accumulation_steps 4 \
  --learning_rate 5e-4 \
  --image_aug False \
  --wandb_project LG_TransferWetTissue \
  --wandb_entity jellyho_ \
  --save_steps 1000 \
  --window_size 1 \
  --future_action_window_size 4