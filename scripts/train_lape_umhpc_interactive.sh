#!/bin/bash -l
# =============================================================================
# LAPE DTU 单卡即时训练: 在已分配的 UMHPC interactive GPU 节点前台运行。
# 不申请资源, 不自动提交或续投作业。训练参数以 train_lape_umhpc.sh 为基准。
#
#   cd <本 checkout 的根目录>
#   bash scripts/train_lape_umhpc_interactive.sh          # 默认 10 epoch, batch 2
#   PER_GPU_BATCH=1 VAL_BATCH_SIZE=1 bash scripts/train_lape_umhpc_interactive.sh
#   LAPE=off bash scripts/train_lape_umhpc_interactive.sh  # DA3 + MoA 对照
#   FRESH=0 RUN_NAME=<原实验名> bash scripts/train_lape_umhpc_interactive.sh
#
# 默认实验名与原脚本一致; 输出位于 log/experiments/$RUN_NAME/。
# 训练参数默认值与 train_lape_umhpc.sh 一致, 包括 STEPS=0 和 WARMUP_STEPS=1000。
# FRESH=1 会归档同名旧 run (保留原脚本的最近写入保护), FRESH=0 自动续训。
# 可设置 PYTHON_BIN=/path/to/uprmvs/bin/python 跳过 conda 激活。
# interactive 分配到期后不会自动续投, 后续续训需指定原 RUN_NAME 和 FRESH=0。
# =============================================================================

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
if [[ ! -f "$PROJECT_DIR/train_moa.py" ]]; then
    echo "PROJECT_DIR=$PROJECT_DIR 不像是本仓库的根目录 (没有 train_moa.py); 请设置 PROJECT_DIR 为仓库根目录" >&2
    exit 2
fi
cd "$PROJECT_DIR"

LAPE=${LAPE:-on}
case "$LAPE" in
    on)  RUN_NAME=${RUN_NAME:-LAPE_DTU_E10} ;;
    off) RUN_NAME=${RUN_NAME:-DA3MOA_DTU_E10} ;;
    *) echo "LAPE 只能是 on / off, 收到 '$LAPE'" >&2; exit 2 ;;
esac
EPOCHS=${EPOCHS:-10}
STEPS=${STEPS:-0}                        # >0 则按步数跑 (覆盖 EPOCHS); 0 = 按 EPOCHS
WARP_CHANNELS=${WARP_CHANNELS:-128,128,128,128}
PER_GPU_BATCH=${PER_GPU_BATCH:-2}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-4}
NUM_VIEWS=${NUM_VIEWS:-5}
NUM_WORKERS=${NUM_WORKERS:-12}
WARMUP_STEPS=${WARMUP_STEPS:-1000}
VAL_INTERVAL=${VAL_INTERVAL:-5000}       # 另外每个 epoch 末都验证一次
CKPT_INTERVAL=${CKPT_INTERVAL:-1000}
LOG_INTERVAL=${LOG_INTERVAL:-20}
SEED=${SEED:-20260526}
DETERMINISTIC=${DETERMINISTIC:-1}
DA3_RES=${DA3_RES:-518}
FRESH=${FRESH:-1}                        # 1 = 新 run (归档同名旧目录, --resume off); 0 = 续训
# lr: 3e-4 @ 全局 batch 2, 按 sqrt 缩放 (与 train_moa_umhpc.sh 同一规则)
LR_REF=${LR_REF:-3e-4}
LR_REF_BATCH=${LR_REF_BATCH:-2}
LR=${LR:-$(awk -v l="$LR_REF" -v g="$PER_GPU_BATCH" -v r="$LR_REF_BATCH" 'BEGIN{printf "%.4g", l*sqrt(g/r)}')}

# 可指定环境里的 Python, 此时不需要 source ~/.bashrc 或 conda activate。
if [[ -z "${PYTHON_BIN:-}" ]]; then
    set +u
    source ~/.bashrc
    conda activate uprmvs
    set -u
    PYTHON_BIN=$(command -v python)
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python 不存在或不可执行: $PYTHON_BIN" >&2
    exit 2
fi

export UPRMVS_MACHINE=umhpc
export UPRMVS_PROFILE=umhpc
export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/models:$PROJECT_DIR/models/Depth-Anything-3/src"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
mkdir -p logs

RUN_DIR="log/experiments/$RUN_NAME"
if [[ "$FRESH" == "1" ]]; then
    if [[ -e "$RUN_DIR" ]]; then
        RECENT=$(find "$RUN_DIR" -type f -mmin -60 -print -quit 2>/dev/null || true)
        if [[ -n "$RECENT" && "${FORCE_ARCHIVE:-0}" != "1" ]]; then
            echo "拒绝归档: $RUN_DIR 最近 60 分钟内仍有写入 ($RECENT); 换 RUN_NAME 或确认结束后 FORCE_ARCHIVE=1" >&2
            exit 2
        fi
        ARCHIVE="log/experiments/_archive/${RUN_NAME}_$(date -u +%Y%m%d_%H%M%S)"
        mkdir -p "$(dirname "$ARCHIVE")"
        mv "$RUN_DIR" "$ARCHIVE"
        echo "=== 上一轮已归档 (未删除): $RUN_DIR -> $ARCHIVE ==="
    fi
    RESUME=off
else
    RESUME=auto
fi

GIT_SHA=$(git rev-parse HEAD 2>/dev/null || echo unknown)
echo "=================================================================="
echo " LAPE DTU interactive  lape=$LAPE  run=$RUN_NAME  job=${SLURM_JOB_ID:-manual}  host=$(hostname)"
echo " git=${GIT_SHA:0:12}  fresh=$FRESH  resume=$RESUME"
echo " epochs=$EPOCHS steps=$STEPS batch=$PER_GPU_BATCH views=$NUM_VIEWS warp=$WARP_CHANNELS da3_res=$DA3_RES"
echo " lr=$LR warmup=$WARMUP_STEPS seed=$SEED"
echo "=================================================================="
nvidia-smi -L || true
"$PYTHON_BIN" - "$PER_GPU_BATCH" <<'PY' || exit 1
import sys, torch
from base.config import ProjectPaths
if not torch.cuda.is_available():
    sys.exit("CUDA 不可用 —— 请在已分配的 interactive GPU 节点中运行")
gib = torch.cuda.get_device_properties(0).total_memory / 2**30
print(f"=== GPU {torch.cuda.get_device_name(0)}  {gib:.0f} GiB ===")
if gib < 70 and int(sys.argv[1]) >= 2:
    sys.exit(f"只分到 {gib:.0f} GiB 的卡, batch {sys.argv[1]} 需要 80GB A100; 换节点或设置 PER_GPU_BATCH=1")
w = ProjectPaths().da3_weights_file
if not (w / "model.safetensors").is_file():
    sys.exit(f"DA3 权重不在 {w} (需要 config.json + model.safetensors)")
import depth_anything_3.api  # noqa: F401  (fail here, not after the dataset is built)
print(f"=== DA3 weights {w} ===")
PY

args=(
    --profile umhpc
    --name "$RUN_NAME"
    --moa on --lape "$LAPE" --feat-backbone da3 --da3-process-res "$DA3_RES"
    --warp-channels "$WARP_CHANNELS"
    --batch-size "$PER_GPU_BATCH"
    --val-batch-size "$VAL_BATCH_SIZE"
    --num-views "$NUM_VIEWS"
    --num-workers "$NUM_WORKERS"
    --lr "$LR"
    --warmup-steps "$WARMUP_STEPS"
    --amp on --amp-dtype bf16
    --multi-scale on
    --seed "$SEED"
    --log-interval "$LOG_INTERVAL"
    --val-interval "$VAL_INTERVAL"
    --ckpt-interval "$CKPT_INTERVAL"
    --resume "$RESUME"
)
if [[ "$STEPS" -gt 0 ]]; then
    args+=(--max-steps "$STEPS")
else
    args+=(--epochs "$EPOCHS")
fi
[[ "$DETERMINISTIC" == "1" ]] && args+=(--deterministic)

exec "$PYTHON_BIN" train_moa.py "${args[@]}"
