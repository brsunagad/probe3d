#!/bin/bash
#SBATCH --job-name=correspondence_eval
#SBATCH --output=logs_teacher/correspondence_%A_%a.out
#SBATCH --error=logs_teacher/correspondence_%A_%a.err
#SBATCH --partition=tflmb_gpu-rtx4090
#SBATCH --gres=gpu:1
#SBATCH --array=0-27
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00

# Activate environment
source ~/.bashrc
micromamba activate probe3d

# Define base checkpoint directory
# BASE_DIR="/work/dlclarge2/sunagadb-runs/dinov2/outputs_b14/eval"
BASE_DIR="/work/dlclarge2/sunagadb-runs/dinov2/outputs_b14/eval"
export HYDRA_FULL_ERROR=1

# Get latest 5 training checkpoint folders sorted numerically
CHECKPOINTS=($(ls -d ${BASE_DIR}/training_* | sort -V | tail -n 18))
# CHECKPOINTS=($(ls -d ${BASE_DIR}/training_* | sort -V | tail -n 5 | head -n 4)) # exclude the last one


# Generate COMMANDS array
COMMANDS=()
for CKPT_PATH in "${CHECKPOINTS[@]}"; do
    CKPT_ID=$(basename "$CKPT_PATH")
    echo "🧪 Evaluating checkpoint: ${CKPT_PATH}"

    # COMMANDS+=("python train_depth.py backbone=custom_dinov2_b14_reg backbone.teacher_checkpoint=${CKPT_PATH}/teacher_checkpoint.pth +backbone.return_multilayer=True")
    COMMANDS+=("python train_snorm.py backbone=custom_dinov2_b14_reg backbone.teacher_checkpoint=${CKPT_PATH}/teacher_checkpoint.pth +backbone.return_multilayer=True")

    # COMMANDS+=("python evaluate_navi_correspondence.py backbone=custom_dinov2_b14_reg backbone.teacher_checkpoint=${CKPT_PATH}/teacher_checkpoint.pth")
    # COMMANDS+=("python evaluate_navi_correspondence.py backbone=custom_dinov2_b14 +backbone.return_multilayer=True multilayer=True backbone.teacher_checkpoint=${CKPT_PATH}/teacher_checkpoint.pth")

    # COMMANDS+=("python evaluate_scannet_correspondence.py backbone=custom_dinov2_b14_reg backbone.teacher_checkpoint=${CKPT_PATH}/teacher_checkpoint.pth")
    # COMMANDS+=("python evaluate_scannet_correspondence.py backbone=custom_dinov2_b14 +backbone.return_multilayer=True multilayer=True backbone.teacher_checkpoint=${CKPT_PATH}/teacher_checkpoint.pth")

    # COMMANDS+=("python evaluate_spair_correspondence.py backbone=custom_dinov2_b14_reg backbone.teacher_checkpoint=${CKPT_PATH}/teacher_checkpoint.pth")
    # COMMANDS+=("python evaluate_spair_correspondence.py backbone=custom_dinov2_b14 +backbone.return_multilayer=True multilayer=True backbone.teacher_checkpoint=${CKPT_PATH}/teacher_checkpoint.pth")
done


BASE_DIR="/work/dlclarge2/sunagadb-runs/dinov2/outputs_b14_long/eval"

# Get latest 5 training checkpoint folders sorted numerically
CHECKPOINTS=($(ls -d ${BASE_DIR}/training_* | sort -V | tail -n 18))



for CKPT_PATH in "${CHECKPOINTS[@]}"; do
    CKPT_ID=$(basename "$CKPT_PATH")
    echo "🧪 Evaluating checkpoint: ${CKPT_PATH}"

    # COMMANDS+=("python train_depth.py backbone=custom_dinov2_b14 backbone.teacher_checkpoint=${CKPT_PATH}/teacher_checkpoint.pth +backbone.return_multilayer=True")
    COMMANDS+=("python train_snorm.py backbone=custom_dinov2_b14 backbone.teacher_checkpoint=${CKPT_PATH}/teacher_checkpoint.pth +backbone.return_multilayer=True")

    # COMMANDS+=("python evaluate_navi_correspondence.py backbone=custom_dinov2_b14_reg backbone.teacher_checkpoint=${CKPT_PATH}/teacher_checkpoint.pth")
    # COMMANDS+=("python evaluate_navi_correspondence.py backbone=custom_dinov2_b14 +backbone.return_multilayer=True multilayer=True backbone.teacher_checkpoint=${CKPT_PATH}/teacher_checkpoint.pth")

    # COMMANDS+=("python evaluate_scannet_correspondence.py backbone=custom_dinov2_b14_reg backbone.teacher_checkpoint=${CKPT_PATH}/teacher_checkpoint.pth")
    # COMMANDS+=("python evaluate_scannet_correspondence.py backbone=custom_dinov2_b14 +backbone.return_multilayer=True multilayer=True backbone.teacher_checkpoint=${CKPT_PATH}/teacher_checkpoint.pth")

    # COMMANDS+=("python evaluate_spair_correspondence.py backbone=custom_dinov2_b14_reg backbone.teacher_checkpoint=${CKPT_PATH}/teacher_checkpoint.pth")
    # COMMANDS+=("python evaluate_spair_correspondence.py backbone=custom_dinov2_b14 +backbone.return_multilayer=True multilayer=True backbone.teacher_checkpoint=${CKPT_PATH}/teacher_checkpoint.pth")
done

# Run the command based on SLURM task ID
eval ${COMMANDS[$SLURM_ARRAY_TASK_ID]}
