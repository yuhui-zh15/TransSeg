#!/bin/bash

#SBATCH --job-name=msd
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4gb
#SBATCH --partition=pasteur
#SBATCH --gres=gpu:4
#SBATCH --time=100:00:00
#SBATCH --output=msd_01_%A_%a.out
#SBATCH --mail-type=ALL

python main.py \
  --data_dir data/msd/processed/Task01_BrainTumour/   \
  --split_json dataset_5slices.json \
  --img_size 240 240 5 \
  --clip_range -1000000 1000000 \
  --in_channels 4 \
  --out_channels 4 \
  --max_steps 250000 \
  --train_batch_size 4 \
  --eval_batch_size 4 \
  --accumulate_grad_batches 1
