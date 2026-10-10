#!/bin/bash -l
#SBATCH --job-name=lapemvs_lape_gru
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --qos=long
#SBATCH --time=3-00:00:00
#SBATCH --signal=B:USR1@900
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# =============================================================================
# LAPE-GRU —— DTU 从头训练 10 个 epoch (双卡 A100-80GB, 单节点 DDP, sbatch 提交)。
#
#   cd <本 checkout 的根目录> && git checkout dev && git pull && mkdir -p logs
#   sbatch scripts/train_lape_gru_umhpc.sh                    # LAPE_GRU_DTU_E10, 每卡 batch 2 / 全局 batch 4
#   EPOCHS=1 RUN_NAME=LAPE_GRU_E1 sbatch scripts/train_lape_gru_umhpc.sh   # 先跑 1 个 epoch 看速度
#
# 相对 LAPE (scripts/train_lape_umhpc.sh) 的改动 (全部是 base/config_moa.py 的默认值, 无需额外参数):
#   * LAPE-GRU (models/moa/affine_gru.py): level 0/1/2 用 ConvGRU 迭代 4/4/3 步更新单目局部仿射
#     的中心化参数 (alpha, b~); 每步用 expert 窗口和做"再拟合"残差 + 在父级 cost volume / 原始
#     后验上沿当前单目深度 lookup; 初值 = 窗口 expert 的逆方差混合, 更新头零初始化。结果作为
#     第 5 个单目专家 E_gru 进入混合头 (6 路) 与下一级先验, 并 lift 到子分辨率 DA3 上;
#   * 末级精修 (models/moa/refine.py): 1/2 分辨率 ConvGRU 4 步, 每步在当前深度附近重做 5 候选
#     plane sweep, 残差凸上采样回全分辨率 -> depth_full; 另输出 conf_refine;
#   * 关闭低频回拉 (lape.lfr=False), 取消单目权重上限 (moa.moa_gain=(1,1,1));
#   * 新损失: GRU 序列损失 + 子分辨率损失 + 边缘感知 TV, 先验混合 NLL, 精修序列损失 + 置信度 BCE。
#   必须从 step 0 训练 (混合头变成 6 路, 旧 LAPE checkpoint 不能续训)。
#
# 双卡: torchrun 每卡一个进程; 每卡 batch 2, 全局 batch 4, lr 3e-4 @ batch 2 按 sqrt 缩放 -> 4.243e-4。
#   NUM_WORKERS=12 是每进程的 worker 数; 验证 batch 4 也是每卡值。速度/显存以日志为准
#   (旧 LAPE 双卡约 3.14 s/step, 10 epoch 67,740 步 ~59 h; GRU + 精修会再慢一些)。
#
# 超时自动续投: slurm 在超时前 15 分钟发 USR1, 所有 rank 跑完当前 step 后同步保存 latest.pth 并退出;
#   脚本识别 stop-file 后以 FRESH=0 重投本脚本。手动续训: FRESH=0 sbatch scripts/train_lape_gru_umhpc.sh
#
# 输出: log/experiments/$RUN_NAME/{model/{latest,best}.pth, tensorboard, config.json}
# 训完: RUN_NAME=LAPE_GRU_DTU_E10 CKPT=log/experiments/LAPE_GRU_DTU_E10/model/latest.pth \
#       sbatch scripts/test_lape_dtu_umhpc.sh
# =============================================================================

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}
if [[ ! -f "$PROJECT_DIR/train_moa.py" ]]; then
    echo "PROJECT_DIR=$PROJECT_DIR 不像是本仓库的根目录 (没有 train_moa.py); 请在仓库根目录 sbatch" >&2
    exit 2
fi
cd "$PROJECT_DIR"

RUN_NAME=${RUN_NAME:-LAPE_GRU_DTU_E10}
EPOCHS=${EPOCHS:-10}
STEPS=${STEPS:-0}                        # >0 则按步数跑 (覆盖 EPOCHS); 0 = 按 EPOCHS
WARP_CHANNELS=${WARP_CHANNELS:-128,128,128,128}
NPROC=2                                 # 单节点双卡, 一卡一个训练进程
PER_GPU_BATCH=${PER_GPU_BATCH:-2}
GLOBAL_BATCH=$((NPROC * PER_GPU_BATCH))
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
CHAIN=${CHAIN:-0}
MAX_CHAIN=${MAX_CHAIN:-4}
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
echo " LAPE-GRU DTU  run=$RUN_NAME  job=${SLURM_JOB_ID:-manual}  host=$(hostname)"
echo " git=${GIT_SHA:0:12}  chain=$CHAIN/$MAX_CHAIN  fresh=$FRESH  resume=$RESUME"
echo " epochs=$EPOCHS steps=$STEPS per_gpu_batch=$PER_GPU_BATCH global_batch=$GLOBAL_BATCH gpus=$NPROC views=$NUM_VIEWS warp=$WARP_CHANNELS da3_res=$DA3_RES"
echo " lr=$LR warmup=$WARMUP_STEPS seed=$SEED"
echo "=================================================================="
nvidia-smi -L || true
python - "$PER_GPU_BATCH" "$NPROC" <<'PY' || exit 1
import sys, torch
from base.config import ProjectPaths
if not torch.cuda.is_available() or torch.cuda.device_count() < int(sys.argv[2]):
    sys.exit("需要两张可见 GPU —— 这个作业需要 --gres=gpu:2")
for i in range(int(sys.argv[2])):
    prop = torch.cuda.get_device_properties(i)
    gib = prop.total_memory / 2**30
    print(f"=== GPU {i}: {prop.name}  {gib:.0f} GiB ===")
    if "A100" not in prop.name or gib < 70:
        sys.exit(f"GPU {i} 是 {prop.name} / {gib:.0f} GiB; 本脚本要求两张 A100 80GB")
w = ProjectPaths().da3_weights_file
if not (w / "model.safetensors").is_file():
    sys.exit(f"DA3 权重不在 {w} (需要 config.json + model.safetensors)")
import depth_anything_3.api  # noqa: F401  (fail here, not after the dataset is built)
print(f"=== DA3 weights {w} ===")
PY

args=(
    --profile umhpc
    --name "$RUN_NAME"
    --moa on --lape on --feat-backbone da3 --da3-process-res "$DA3_RES"
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
STOP_FILE=$(mktemp "$PROJECT_DIR/logs/lape_gru_stop_${SLURM_JOB_ID:-manual}.XXXXXX")
rm -f "$STOP_FILE"
trap 'echo "=== 收到超时信号, 请求所有 rank 存档 ==="; touch "$STOP_FILE"' USR1 TERM
python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node="$NPROC" --max-restarts=0 \
    train_moa.py "${args[@]}" --stop-file "$STOP_FILE" &
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
    sbatch --export=ALL,FRESH=0,CHAIN=$NEXT,RUN_NAME=$RUN_NAME,PER_GPU_BATCH=$PER_GPU_BATCH,LR=$LR,EPOCHS=$EPOCHS,STEPS=$STEPS,WARP_CHANNELS=$WARP_CHANNELS,DA3_RES=$DA3_RES \
        "$PROJECT_DIR/scripts/train_lape_gru_umhpc.sh"
    exit 0
fi
echo "=== train_moa.py 退出码 $RC ==="
exit $RC
