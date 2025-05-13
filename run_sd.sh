#!/bin/bash
#SBATCH --job-name=correspondence_eval
#SBATCH --output=logs_sd/correspondence_%A_%a.out
#SBATCH --error=logs_sd/correspondence_%A_%a.err
#SBATCH --array=0
#SBATCH --partition=tflmb_gpu-rtx3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00

# Activate environment
source ~/.bashrc
micromamba activate probe3d

COMMANDS=(
    # "python train_depth.py backbone=dinov2_b14 +backbone.return_multilayer=True"
    # "python train_snorm.py backbone=dinov2_b14 +backbone.return_multilayer=True"

    # "python train_depth.py backbone=custom_dinov2_b14_reg +backbone.return_multilayer=True"
    # "python train_snorm.py backbone=custom_dinov2_b14_reg +backbone.return_multilayer=True"

    # "python train_depth.py backbone=dino_b16 +backbone.return_multilayer=True"
    "python train_snorm.py backbone=dino_b16 +backbone.return_multilayer=True"


)
# Run the command corresponding to the array task ID
eval ${COMMANDS[$SLURM_ARRAY_TASK_ID]}
