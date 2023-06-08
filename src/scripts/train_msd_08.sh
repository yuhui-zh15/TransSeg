#!/bin/bash

#SBATCH --job-name=msd
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=4gb
#SBATCH --partition=pasteur
#SBATCH --gres=gpu:4
#SBATCH --time=100:00:00
#SBATCH --output=msd_08_%A_%a.out
#SBATCH --mail-type=ALL

python main.py \
  --data_dir data/msd/processed/Task08_HepaticVessel/   \
  --split_json dataset_5slices.json \
  --img_size 512 512 5 \
  --clip_range -175 250 \
  --in_channels 1 \
  --out_channels 3 \
  --max_steps 100000 \
  --train_batch_size 2 \
  --eval_batch_size 2 \
  --accumulate_grad_batches 2
