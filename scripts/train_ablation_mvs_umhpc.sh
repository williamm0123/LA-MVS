#!/bin/bash -l
#SBATCH --job-name=lapemvs_abl_mvs
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --qos=long
#SBATCH --time=3-00:00:00
#SBATCH --signal=B:USR1@900
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# =============================================================================
# 消融: 纯 MVS (去掉单目先验) —— DTU 从头训练 10 个 epoch (umhpc 单卡 A100-80GB, sbatch 提交)。
#
#   cd <本 checkout 的根目录> && git checkout dev && git pull && mkdir -p logs
#   sbatch scripts/train_ablation_mvs_umhpc.sh                                  # ABL_MVS_DA3_E10
#   VARIANT=mvs_fpn sbatch scripts/train_ablation_mvs_umhpc.sh                  # ABL_MVS_FPN_E10
#   EPOCHS=1 RUN_NAME=ABL_MVS_DA3_E1 sbatch scripts/train_ablation_mvs_umhpc.sh # 先跑 1 epoch 看速度
#
# 入口 ablation.py (训练循环 = train_moa.run, 只固定结构):
#   VARIANT=mvs_da3 (默认): DA3 编码器 token -> SVA -> FPN + 四级 cascade; 无 MoA/LAPE/LAPE-GRU、
#       无先验融合、无 1/2 分辨率精修, DA3 的 DPT 头不运行 —— 与 LAPE-GRU 只差单目先验;
#   VARIANT=mvs_fpn: 再去掉冻结 ViT (feat.backbone=none), 纯 FPN 匹配网络。
#   其余超参与 scripts/train_lape_gru_umhpc.sh 完全一致 (batch 2, lr 3e-4, warp 128x4,
#   multi-scale, bf16, 同一 seed), 损失只剩各级 CE。
#
# 超时自动续投: slurm 在超时前 15 分钟发 USR1, 保存 latest.pth 后以 FRESH=0 重投本脚本。
#   手动续训: FRESH=0 RUN_NAME=<同名> sbatch scripts/train_ablation_mvs_umhpc.sh
#
# 输出: log/experiments/$RUN_NAME/{model/{latest,best}.pth, tensorboard, config.json}
# 训完: RUN_NAME=ABL_MVS_DA3_E10 CKPT=log/experiments/ABL_MVS_DA3_E10/model/latest.pth \
#       sbatch scripts/test_lape_dtu_umhpc.sh
# =============================================================================

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}
if [[ ! -f "$PROJECT_DIR/ablation.py" ]]; then
    echo "PROJECT_DIR=$PROJECT_DIR 不像是本仓库的根目录 (没有 ablation.py); 请在仓库根目录 sbatch" >&2
    exit 2
fi
cd "$PROJECT_DIR"

VARIANT=${VARIANT:-mvs_da3}               # mvs_da3 | mvs_fpn
case "$VARIANT" in
    mvs_da3) DEFAULT_NAME=ABL_MVS_DA3_E10 ;;
    mvs_fpn) DEFAULT_NAME=ABL_MVS_FPN_E10 ;;
    *) echo "VARIANT=$VARIANT 不认识 (mvs_da3 | mvs_fpn)" >&2; exit 2 ;;
esac
RUN_NAME=${RUN_NAME:-$DEFAULT_NAME}
EPOCHS=${EPOCHS:-10}
STEPS=${STEPS:-0}                        # >0 则按步数跑 (覆盖 EPOCHS); 0 = 按 EPOCHS
WARP_CHANNELS=${WARP_CHANNELS:-128,128,128,128}
NPROC=1                                 # 单卡
PER_GPU_BATCH=${PER_GPU_BATCH:-2}
GLOBAL_BATCH=$((NPROC * PER_GPU_BATCH))
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-4}
NUM_VIEWS=${NUM_VIEWS:-5}
NUM_WORKERS=${NUM_WORKERS:-8}
WARMUP_STEPS=${WARMUP_STEPS:-1000}
VAL_INTERVAL=${VAL_INTERVAL:-5000}       # 另外每个 epoch 末都验证一次
CKPT_INTERVAL=${CKPT_INTERVAL:-1000}
LOG_INTERVAL=${LOG_INTERVAL:-20}
SEED=${SEED:-20260526}
DETERMINISTIC=${DETERMINISTIC:-1}
DA3_RES=${DA3_RES:-518}
FRESH=${FRESH:-1}                        # 1 = 新 run (归档同名旧目录, --resume off); 0 = 续训
CHAIN=${CHAIN:-0}
MAX_CHAIN=${MAX_CHAIN:-6}
# lr: 3e-4 @ 全局 batch 2, 按 sqrt 缩放 (与 train_moa_umhpc.sh 同一规则)
LR_REF=${LR_REF:-3e-4}
LR_REF_BATCH=${LR_REF_BATCH:-2}
LR=${LR:-$(awk -v l="$LR_REF" -v g="$GLOBAL_BATCH" -v r="$LR_REF_BATCH" 'BEGIN{printf "%.4g", l*sqrt(g/r)}')}

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
echo " 纯 MVS 消融 ($VARIANT) DTU  run=$RUN_NAME  job=${SLURM_JOB_ID:-manual}  host=$(hostname)"
echo " git=${GIT_SHA:0:12}  chain=$CHAIN/$MAX_CHAIN  fresh=$FRESH  resume=$RESUME"
echo " epochs=$EPOCHS steps=$STEPS per_gpu_batch=$PER_GPU_BATCH global_batch=$GLOBAL_BATCH gpus=$NPROC views=$NUM_VIEWS warp=$WARP_CHANNELS da3_res=$DA3_RES"
echo " lr=$LR warmup=$WARMUP_STEPS seed=$SEED"
echo "=================================================================="
nvidia-smi -L || true
python - "$PER_GPU_BATCH" "$NPROC" "$VARIANT" <<'PY' || exit 1
import sys, torch
from base.config import ProjectPaths
if not torch.cuda.is_available() or torch.cuda.device_count() < int(sys.argv[2]):
    sys.exit("没有可见 GPU —— 这个作业需要 --gres=gpu:1")
for i in range(int(sys.argv[2])):
    prop = torch.cuda.get_device_properties(i)
    gib = prop.total_memory / 2**30
    print(f"=== GPU {i}: {prop.name}  {gib:.0f} GiB ===")
    if "A100" not in prop.name or gib < 70:
        sys.exit(f"GPU {i} 是 {prop.name} / {gib:.0f} GiB; 本脚本要求 A100 80GB")
if sys.argv[3] == "mvs_fpn":
    sys.exit(0)                         # 不用 DA3
w = ProjectPaths().da3_weights_file
if not (w / "model.safetensors").is_file():
    sys.exit(f"DA3 权重不在 {w} (需要 config.json + model.safetensors)")
import depth_anything_3.api  # noqa: F401  (fail here, not after the dataset is built)
print(f"=== DA3 weights {w} ===")
PY

args=(
    --profile umhpc
    --name "$RUN_NAME"
    --variant "$VARIANT" --da3-process-res "$DA3_RES"
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

# 不向 torchrun 父进程发 USR1 (会直接终止 launcher)。共享文件让所有 rank
# 在完成当前训练 step 后同步存档; launcher 正常退出后再触发原自动续投逻辑。
STOP_FILE=$(mktemp "$PROJECT_DIR/logs/abl_mvs_stop_${SLURM_JOB_ID:-manual}.XXXXXX")
rm -f "$STOP_FILE"
trap 'echo "=== 收到超时信号, 请求所有 rank 存档 ==="; touch "$STOP_FILE"' USR1 TERM
python ablation.py --device cuda:0 "${args[@]}" --stop-file "$STOP_FILE" &
PID=$!
set +e
wait "$PID"
RC=$?
while [[ $RC -gt 128 ]] && kill -0 "$PID" 2>/dev/null; do
    wait "$PID"
    RC=$?
done
set -e
if [[ $RC -eq 0 && -f "$STOP_FILE" ]]; then
    RC=124
fi
rm -f "$STOP_FILE"
trap - USR1 TERM

if [[ $RC -eq 124 ]]; then
    if [[ $CHAIN -ge $MAX_CHAIN ]]; then
        echo "=== 已续投 $CHAIN 次 (MAX_CHAIN=$MAX_CHAIN), 不再自动续投; 手动: FRESH=0 sbatch $0 ===" >&2
        exit 1
    fi
    NEXT=$((CHAIN + 1))
    echo "=== 超时存档完成, 续投第 $NEXT 次 ==="
    sbatch --export=ALL,FRESH=0,CHAIN=$NEXT,VARIANT=$VARIANT,RUN_NAME=$RUN_NAME,PER_GPU_BATCH=$PER_GPU_BATCH,LR=$LR,EPOCHS=$EPOCHS,STEPS=$STEPS,WARP_CHANNELS=$WARP_CHANNELS,DA3_RES=$DA3_RES \
        "$PROJECT_DIR/scripts/train_ablation_mvs_umhpc.sh"
    exit 0
fi
echo "=== ablation.py 退出码 $RC ==="
exit $RC
