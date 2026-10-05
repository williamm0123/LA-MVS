#!/bin/bash -l
#SBATCH --job-name=lapemvs_tnt
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --qos=long
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# =============================================================================
# Tanks-and-Temples 推理 + 融合 (MoAMVSNet) —— 单卡 A100, sbatch 提交。
#
#   cd <lapemvs 在 umhpc 的 checkout> && git pull && mkdir -p logs
#   sbatch scripts/test_tnt_umhpc.sh                                    # LAPE BlendedMVS 微调后的 latest.pth
#   CKPT=log/experiments/LAPE_DTU_E10/model/latest.pth TAG=LAPE_DTU_E10 sbatch scripts/test_tnt_umhpc.sh   # 对照: 纯 DTU 权重
#   CKPT=log/experiments/MOA1_BLD_30K/model/latest.pth TAG=MOA1_BLD sbatch scripts/test_tnt_umhpc.sh      # 旧 moa1
#   SCENES="intermediate/Family advanced/Temple" sbatch ...             # 子集
#   PHASE=fuse CONF=0.3 PLY_DIR=log/tnt/xxx/ply_c03 sbatch ...          # 只换阈值重融
#
# 步骤:
#   1. DA3 缓存 log/da3_cache_tnt —— 只有旧 (DINOv3) checkpoint 需要; DA3 特征的新模型 (LAPE) 在网络里
#      在线推理 DA3, 自动跳过这一步 (按 checkpoint 快照里的 feat.backbone 判断)
#   2. test_tt_moa.py: 1920x1080 整幅, NUM_VIEWS 个视角 (默认 7; 本地实测 7 视角 720x1280 峰值
#      12.6 GiB, 1080p 外推约 28 GiB), 逐视角缓存 -> MonoMVSNet 动态几何一致性融合
#   输出: $OUT/depth/<split>/<Scene>/*.npz,  $PLY_DIR/<Scene>.ply (默认 $OUT/ply)
#   没有本地 GT: 把 <Scene>.ply 和数据集自带的 <Scene>.log 一起上传 T&T 官网打分。
#
# 融合阈值是 DTU 的 (stage4 最大后验 > 0.55)。本地 Family 0.5x 实测只有 7~9% 像素过这道门,
# 点云会偏稀; 4 个假设的最大后验下限是 0.25, 本地 conf_last 中位数约 0.41, 0.3 已近乎不过滤。
# 建议推理完用 PHASE=fuse 扫 CONF (0.4 / 0.45 / 0.5 / 0.55) 各出一份再挑。
# 中断后原样重投: 已有的逐视角 NPZ 和 PLY 会跳过。
# =============================================================================

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}
[[ -f "$PROJECT_DIR/test_tt_moa.py" ]] || { echo "PROJECT_DIR=$PROJECT_DIR 不是仓库根目录" >&2; exit 2; }
cd "$PROJECT_DIR"

CKPT=${CKPT:-log/experiments/LAPE_BLD_E10/model/latest.pth}
TAG=${TAG:-$(basename "$(dirname "$(dirname "$CKPT")")")}
TNT_ROOT=${TNT_ROOT:-/scr/user/qinglong/dataset/TankandTemples}
SCENES=${SCENES:-}                      # 空 = TNT_ROOT 下找得到的全部 14 个
NUM_VIEWS=${NUM_VIEWS:-7}
RESIZE=${RESIZE:-1.0}
DA3_ROOT=${DA3_ROOT:-$PROJECT_DIR/log/da3_cache_tnt}
DA3_SHARDS=${DA3_SHARDS:-3}
OUT=${OUT:-log/tnt/${TAG}_v${NUM_VIEWS}_r${RESIZE}}
PLY_DIR=${PLY_DIR:-$OUT/ply}
PHASE=${PHASE:-all}
CONF=${CONF:-0.55}
FUSE_WORKERS=${FUSE_WORKERS:-3}         # 每个 worker 整场景驻留内存 (Palace ~16 GB)

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

[[ -d "$TNT_ROOT" ]] || { echo "TNT_ROOT 不存在: $TNT_ROOT (用 TNT_ROOT=... sbatch 指定)" >&2; exit 2; }
if [[ "$PHASE" != "fuse" ]]; then
    [[ -f "$CKPT" ]] || { echo "CKPT 不存在: $CKPT" >&2; exit 2; }
fi
SCENE_ARGS=()
[[ -n "$SCENES" ]] && SCENE_ARGS=(--scenes $SCENES)

echo "=================================================================="
echo " T&T  tag=$TAG  job=${SLURM_JOB_ID:-manual}  host=$(hostname)  git=$(git rev-parse --short=12 HEAD 2>/dev/null || echo unknown)"
echo " ckpt=$CKPT  phase=$PHASE"
echo " tnt=$TNT_ROOT  scenes=${SCENES:-all}  views=$NUM_VIEWS  resize=$RESIZE  conf=$CONF"
echo " out=$OUT  ply=$PLY_DIR  da3=$DA3_ROOT"
echo "=================================================================="
nvidia-smi -L || true

NEEDS_DA3_CACHE=0
if [[ "$PHASE" != "fuse" ]]; then
    python -c 'import sys, torch; sys.exit(0 if torch.cuda.is_available() else "CUDA 不可用 —— 需要 --gres=gpu:1")'
    NEEDS_DA3_CACHE=$(python - "$CKPT" <<'PY'
import sys, torch
arch = torch.load(sys.argv[1], map_location="cpu", weights_only=False).get("arch", {})
print(0 if arch.get("feat", {}).get("backbone") == "da3" else 1)
PY
)
    echo "=== DA3: $([[ $NEEDS_DA3_CACHE == 1 ]] && echo '旧模型, 用离线缓存' || echo '网络内在线推理, 跳过缓存') ==="
fi
if [[ "$NEEDS_DA3_CACHE" == "1" ]]; then
    # ---- 1. DA3 cache (native 1920; resumable) --------------------------------
    DA3_ARGS=(--dataset tnt --root "$TNT_ROOT" --out "$DA3_ROOT" ${SCENE_ARGS[@]+"${SCENE_ARGS[@]}"})
    python scripts/build_da3_cache_mvs.py "${DA3_ARGS[@]}" --dry-run
    pids=()
    for ((i = 0; i < DA3_SHARDS; i++)); do
        python scripts/build_da3_cache_mvs.py "${DA3_ARGS[@]}" --shard "$i/$DA3_SHARDS" \
            > "logs/da3_tnt_${SLURM_JOB_ID:-manual}_$i.log" 2>&1 &
        pids+=($!)
    done
    fail=0
    for p in "${pids[@]}"; do wait "$p" || fail=1; done
    tail -n 2 logs/da3_tnt_${SLURM_JOB_ID:-manual}_*.log || true
    [[ $fail == 0 ]] || { echo "DA3 缓存有分片失败, 见 logs/da3_tnt_${SLURM_JOB_ID:-manual}_*.log" >&2; exit 1; }
fi

# ---- 2. inference + fusion ----------------------------------------------------
args=(--phase "$PHASE" --tt-root "$TNT_ROOT" --out "$OUT" --ply-dir "$PLY_DIR"
      --num-views "$NUM_VIEWS" --resize-scale "$RESIZE" --da3-root "$DA3_ROOT"
      --conf "$CONF" --workers "$FUSE_WORKERS" --profile umhpc ${SCENE_ARGS[@]+"${SCENE_ARGS[@]}"})
[[ "$PHASE" != "fuse" ]] && args+=(--ckpt "$CKPT")
python test_tt_moa.py "${args[@]}"
echo "=== done -> $PLY_DIR ==="
ls -la "$PLY_DIR"
