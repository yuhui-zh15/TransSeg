#!/bin/bash

#SBATCH --job-name=msd
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4gb
#SBATCH --partition=pasteur
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --output=msd_05_%A_%a.out
#SBATCH --mail-type=ALL

python main.py \
  --data_dir data/msd/processed/Task05_Prostate/   \
  --split_json dataset_5slices.json \
  --img_size 320 320 5 \
  --clip_range 0 2152 \
  --in_channels 2 \
  --out_channels 3 \
  --max_steps 25000 \
  --train_batch_size 4 \
  --eval_batch_size 4 \
  --accumulate_grad_batches 1 \
  --force_2d 1