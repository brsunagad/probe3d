#!/bin/bash
#SBATCH --job-name=correspondence_eval
#SBATCH --output=logs/correspondence_%A_%a.out
#SBATCH --error=logs/correspondence_%A_%a.err
#SBATCH --array=0-17
#SBATCH --partition=tflmb_gpu-rtx4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00

# Activate environment
source ~/.bashrc
micromamba activate probe3d

COMMANDS=(

    # ========================
    # NAVI Dataset
    # ========================

    # dino_b16
    "python evaluate_navi_correspondence.py backbone=dino_b16"
    "python evaluate_navi_correspondence.py backbone=dino_b16 +backbone.return_multilayer=True multilayer=True"

    # dinov2_b14
    "python evaluate_navi_correspondence.py backbone=dinov2_b14"
    "python evaluate_navi_correspondence.py backbone=dinov2_b14 +backbone.return_multilayer=True multilayer=True"

    # custom_dinov2_b14_reg
    "python evaluate_navi_correspondence.py backbone=custom_dinov2_b14_reg"
    "python evaluate_navi_correspondence.py backbone=custom_dinov2_b14_reg +backbone.return_multilayer=True multilayer=True"

    # ========================
    # SCANNET Dataset
    # ========================

    # dino_b16
    "python evaluate_scannet_correspondence.py backbone=dino_b16"
    "python evaluate_scannet_correspondence.py backbone=dino_b16 +backbone.return_multilayer=True multilayer=True"

    # dinov2_b14
    "python evaluate_scannet_correspondence.py backbone=dinov2_b14"
    "python evaluate_scannet_correspondence.py backbone=dinov2_b14 +backbone.return_multilayer=True multilayer=True"

    # custom_dinov2_b14_reg
    "python evaluate_scannet_correspondence.py backbone=custom_dinov2_b14_reg"
    "python evaluate_scannet_correspondence.py backbone=custom_dinov2_b14_reg +backbone.return_multilayer=True multilayer=True"

    # ========================
    # SPAIR Dataset
    # ========================

    # dino_b16
    "python evaluate_spair_correspondence.py backbone=dino_b16"
    "python evaluate_spair_correspondence.py backbone=dino_b16 +backbone.return_multilayer=True multilayer=True"

    # dinov2_b14
    "python evaluate_spair_correspondence.py backbone=dinov2_b14"
    "python evaluate_spair_correspondence.py backbone=dinov2_b14 +backbone.return_multilayer=True multilayer=True"

    # custom_dinov2_b14_reg
    "python evaluate_spair_correspondence.py backbone=custom_dinov2_b14_reg"
    "python evaluate_spair_correspondence.py backbone=custom_dinov2_b14_reg +backbone.return_multilayer=True multilayer=True"
)

# Run the command corresponding to the array task ID
eval ${COMMANDS[$SLURM_ARRAY_TASK_ID]}
