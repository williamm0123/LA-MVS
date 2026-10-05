#!/bin/bash -l
#SBATCH --job-name=lapemvs_lape_dtu
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --qos=long
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# =============================================================================
# LAPE 的 DTU 测试: test_moa.py 推理 (DA3 在线, 不需要缓存) + MonoMVSNet 动态几何一致性融合
# (test_dtu.py, 纯 Python, 不需要 fusibile)。打分仍是独立的第三步 (Fast-DTU-Evaluation)。
#
#   sbatch scripts/test_lape_dtu_umhpc.sh                                  # LAPE_DTU_E10/model/best.pth
#   CKPT=log/experiments/LAPE_DTU_E10/model/latest.pth TAG=LAPE_DTU_E10_latest sbatch scripts/test_lape_dtu_umhpc.sh
#   PHASE=fuse sbatch scripts/test_lape_dtu_umhpc.sh                       # 只重融 (换阈值不必重推理)
#
# 口径与 moa1 / moa2 的点云相同: 0.8 整幅 (960x1280), 5 视角, conf_last > 0.55 + 动态几何一致性。
# 输出: $OUT/depth/<scan>/*.npz + metrics.json + run_manifest.json,  $PLY_DIR/mvsnet<scan>_l3.ply
# 中断后原样重投: 已有的逐视角 NPZ 和逐 scan PLY 会跳过 (要全部重算: NO_SKIP=1)。
# =============================================================================

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}
[[ -f "$PROJECT_DIR/test_dtu.py" ]] || { echo "PROJECT_DIR=$PROJECT_DIR 不是仓库根目录" >&2; exit 2; }
cd "$PROJECT_DIR"

RUN_NAME=${RUN_NAME:-LAPE_DTU_E10}
CKPT=${CKPT:-log/experiments/$RUN_NAME/model/best.pth}
TAG=${TAG:-$RUN_NAME}
PHASE=${PHASE:-all}
NUM_VIEWS=${NUM_VIEWS:-5}
RESIZE=${RESIZE:-0.8}
NUM_WORKERS=${NUM_WORKERS:-8}
FUSE_WORKERS=${FUSE_WORKERS:-8}
OUT=${OUT:-log/depth_cache/${TAG}_r${RESIZE}_v${NUM_VIEWS}_test}
PLY_DIR=${PLY_DIR:-log/pred_points_${TAG}_r${RESIZE}_v${NUM_VIEWS}_dypcd}
NO_SKIP=${NO_SKIP:-0}

set +u
source ~/.bashrc
conda activate uprmvs
set -u

export UPRMVS_MACHINE=umhpc
export UPRMVS_PROFILE=umhpc
export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/models:$PROJECT_DIR/models/Depth-Anything-3/src"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
mkdir -p logs

if [[ "$PHASE" != "fuse" ]]; then
    [[ -f "$CKPT" ]] || { echo "找不到 checkpoint: $CKPT (用 CKPT=... 或 RUN_NAME=... 指定)" >&2; exit 1; }
fi
echo "======================================================================"
echo " LAPE DTU test  phase=$PHASE  ckpt=$CKPT  job=${SLURM_JOB_ID:-manual}  host=$(hostname)"
echo " resize=$RESIZE full_image=1 views=$NUM_VIEWS  git=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo " depth cache -> $OUT    ply -> $PLY_DIR"
echo "======================================================================"
nvidia-smi -L || true

args=(--phase "$PHASE" --out "$OUT" --ply-dir "$PLY_DIR" --workers "$FUSE_WORKERS")
[[ "$NO_SKIP" == "1" ]] && args+=(--no-skip-existing)
if [[ "$PHASE" != "fuse" ]]; then
    # 以下参数 test_dtu.py 原样转给 test_moa.py
    args+=(--ckpt "$CKPT" --profile umhpc --split test --full-image --resize-scale "$RESIZE"
           --num-views "$NUM_VIEWS" --num-workers "$NUM_WORKERS")
fi
python test_dtu.py "${args[@]}"
echo "=== done -> $PLY_DIR ==="
ls "$PLY_DIR" | head -30
