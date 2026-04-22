#!/bin/bash
# Resume the active 80x45 third-person RGBD distillation job from its latest checkpoint.
# Submit manually when ready from the RTX 6000 clone:
#   cd /move/u/tylerlum/github_repos/depthbasedRL_rtx6000
#   sbatch scripts/cluster/sbatch_resume_img80_rgbd_third_rtx6000.sh

#SBATCH --job-name=img80_rgbd_third_u1_resume
#SBATCH --partition=move
#SBATCH --nodelist=move5
#SBATCH --account=move
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=/move/u/tylerlum/github_repos/depthbasedRL_rtx6000/slurm_logs/%x_%j.out
#SBATCH --error=/move/u/tylerlum/github_repos/depthbasedRL_rtx6000/slurm_logs/%x_%j.out

set -euo pipefail

cd /move/u/tylerlum/github_repos/depthbasedRL_rtx6000

export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_API_KEY="$(cat /juno/u/tylerlum/.wandb_api_key)"
export WANDB_ENTITY=tylerlum
export WANDB_PROJECT=depthbasedRL-isaacsim-distill

export ISAACSIM_CACHE_ROOT="/tmp/${USER}/isaacsim_cache_${SLURM_JOB_ID}"
export PIP_CACHE_DIR="/tmp/${USER}/pip_cache_${SLURM_JOB_ID}"
export XDG_CACHE_HOME="/tmp/${USER}/xdg_cache_${SLURM_JOB_ID}"
export OMNI_USER_DIR="/tmp/${USER}/omni_user_${SLURM_JOB_ID}"
export OMNI_CACHE_DIR="/tmp/${USER}/omni_cache_${SLURM_JOB_ID}"
mkdir -p "$ISAACSIM_CACHE_ROOT" "$PIP_CACHE_DIR" "$XDG_CACHE_HOME" "$OMNI_USER_DIR" "$OMNI_CACHE_DIR" slurm_logs

nvidia-smi || true
nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu --format=csv -l 60 \
  > "slurm_logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_nvidia_smi.csv" &
SMI_PID=$!
trap 'kill ${SMI_PID} 2>/dev/null || true' EXIT

./scripts/run_in_isaacsim_env.sh python isaacsim_conversion/distill.py \
  --mode train_online --headless --task_source dextoolbench \
  --object_category hammer --object_name claw_hammer --task_name swing_down \
  --student_input camera --student_arch mono_transformer_recurrent \
  --student_checkpoint distillation_runs/img80_rgbd_third_u1_15209713_1536env/checkpoints/student_latest.pt \
  --num_envs 1536 --env_spacing 4.0 --ground_plane_size 500 --camera_backend tiled \
  --distill_config isaacsim_conversion/configs/hammer_distill_rgbd_80x45_window_online_dagger_512.yaml \
  --camera_config isaacsim_conversion/configs/hammer_camera_rgbd_80x45.yaml \
  --teacher_checkpoint pretrained_policy/model.pth --teacher_config pretrained_policy/config.yaml \
  --run_dir "distillation_runs/img80_rgbd_third_u1_resume_${SLURM_JOB_ID}" \
  --wandb --wandb_entity tylerlum --wandb_project depthbasedRL-isaacsim-distill \
  --wandb_name "img80_rgbd_third_u1_resume_${SLURM_JOB_ID}" \
  --capture_viewer --capture_viewer_video --capture_viewer_len 600 --capture_viewer_interval 1000 \
  --capture_viewer_env_id 0 --capture_viewer_wandb_key interactive_viewer \
  --capture_viewer_video_wandb_key policy_input_video --capture_viewer_video_fps 60
