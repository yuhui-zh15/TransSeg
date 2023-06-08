#!/bin/bash

#SBATCH --job-name=msd
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4gb
#SBATCH --partition=pasteur
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --output=msd_04_%A_%a.out
#SBATCH --mail-type=ALL

python main.py \
  --data_dir data/msd/processed/Task04_Hippocampus/   \
  --split_json dataset_5slices.json \
  --img_size 240 240 5 \
  --clip_range 29 205431 \
  --in_channels 1 \
  --out_channels 3 \
  --max_steps 25000 \
  --train_batch_size 4 \
  --eval_batch_size 4 \
  --accumulate_grad_batches 1