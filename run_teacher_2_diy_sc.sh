#!/bin/bash
#SBATCH --job-name=eval_iter
#SBATCH --output=logs_diy_sc/eval_iter_%A_%a.out
#SBATCH --error=logs_diy_sc/eval_iter_%A_%a.err
#SBATCH --partition=lmbdlc2_gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --array=0-4
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00


# =================================

source ~/.bashrc
micromamba activate probe3d

# Enable full Hydra error tracing
export HYDRA_FULL_ERROR=1

COMMANDS=(

    "python train_depth.py backbone=dinov2_b14_diy_sc +backbone.return_multilayer=True"
    "python train_snorm.py backbone=dinov2_b14_diy_sc +backbone.return_multilayer=True"

    "python evaluate_navi_correspondence.py backbone=dinov2_b14_diy_sc"
    # "python evaluate_navi_correspondence.py backbone=dinov2_b14_diy_sc +backbone.return_multilayer=True multilayer=True backbone.teacher_checkpoint=${CKPT_PATH}"

    "python evaluate_scannet_correspondence.py backbone=dinov2_b14_diy_sc"
    # "python evaluate_scannet_correspondence.py backbone=dinov2_b14_diy_sc +backbone.return_multilayer=True multilayer=True backbone.teacher_checkpoint=${CKPT_PATH}"

    "python evaluate_spair_correspondence.py backbone=dinov2_b14_diy_sc"
    # "python evaluate_spair_correspondence.py backbone=dinov2_b14_diy_sc +backbone.return_multilayer=True multilayer=True backbone.teacher_checkpoint=${CKPT_PATH}"

)

# Log which checkpoint is used
eval ${COMMANDS[$SLURM_ARRAY_TASK_ID]}
