#!/bin/bash
#SBATCH --job-name=eval_iter
#SBATCH --output=logs_teacher/eval_iter_%A_%a.out
#SBATCH --error=logs_teacher/eval_iter_%A_%a.err
#SBATCH --partition=tflmb_gpu-rtx3090
#SBATCH --gres=gpu:1
#SBATCH --array=0-4
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00

# ========= MODIFY THIS ===========
ITER=124999  # <<-- set your desired checkpoint iteration here
BASE_DIR="/work/dlclarge2/sunagadb-runs/dinov2/outputs_b14/eval"
CKPT_PATH="${BASE_DIR}/training_${ITER}/teacher_checkpoint.pth"
# =================================

source ~/.bashrc
micromamba activate probe3d

# Enable full Hydra error tracing
export HYDRA_FULL_ERROR=1

COMMANDS=(

    "python train_depth.py backbone=custom_dinov2_b14_reg backbone.teacher_checkpoint=${CKPT_PATH} +backbone.return_multilayer=True"
    "python train_snorm.py backbone=custom_dinov2_b14_reg backbone.teacher_checkpoint=${CKPT_PATH} +backbone.return_multilayer=True"

    "python evaluate_navi_correspondence.py backbone=custom_dinov2_b14_reg backbone.teacher_checkpoint=${CKPT_PATH}"
    # "python evaluate_navi_correspondence.py backbone=custom_dinov2_b14_reg +backbone.return_multilayer=True multilayer=True backbone.teacher_checkpoint=${CKPT_PATH}"

    "python evaluate_scannet_correspondence.py backbone=custom_dinov2_b14_reg backbone.teacher_checkpoint=${CKPT_PATH}"
    # "python evaluate_scannet_correspondence.py backbone=custom_dinov2_b14_reg +backbone.return_multilayer=True multilayer=True backbone.teacher_checkpoint=${CKPT_PATH}"

    "python evaluate_spair_correspondence.py backbone=custom_dinov2_b14_reg backbone.teacher_checkpoint=${CKPT_PATH}"
    # "python evaluate_spair_correspondence.py backbone=custom_dinov2_b14_reg +backbone.return_multilayer=True multilayer=True backbone.teacher_checkpoint=${CKPT_PATH}"

)

# Log which checkpoint is used
echo "🧪 Evaluating checkpoint: ${CKPT_PATH}"
eval ${COMMANDS[$SLURM_ARRAY_TASK_ID]}
