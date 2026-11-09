#!/bin/bash -l
#SBATCH --job-name=lapemvs_abl_test
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
# 消融模型的 DTU 测试 (一张 A100-80GB): 推理 -> 融合 -> [打分]。训练作业结束时会自动投它。
# 口径与 E10 / moa1 完全相同: 22 个测试 scan, 0.8 整幅 (960x1280), 5 视角, bf16,
#   conf_last > 0.55 + 动态几何一致性融合 (test_dtu.py), Fast-DTU-Evaluation 打分。
#
#   RUN_NAME=ABL_A0_E10 sbatch scripts/ablation/test_umhpc.sh
#   RUN_NAME=LAPE_GRU_DTU_E10 sbatch scripts/ablation/test_umhpc.sh      # FULL 的 seed 1
#   PHASE=fuse RUN_NAME=... sbatch scripts/ablation/test_umhpc.sh        # 只重融
#   PHASE=score RUN_NAME=... bash scripts/ablation/test_umhpc.sh         # 只打分 (可在本地跑)
#
# CKPT 默认 latest.pth (退火到底的那一份; E10 论文数字也是 latest), 想用 best: CKPT_KIND=best
# 输出 (scripts/ablation/collect.py 从这里读):
#   log/ablation/$RUN_NAME/depth_cache/metrics.json   深度指标 + window (Cov_2..4, E_in/E_out)
#   log/ablation/$RUN_NAME/ply_dypcd/*.ply            融合点云
#   log/ablation/$RUN_NAME/ply_dypcd/fast_dtu_eval.log  Acc/Comp/Overall (打分后)
# 打分需要 Fast-DTU-Evaluation + DTU SampleSet GT: 设 FAST_DTU_DIR / DTU_GT_DIR; 不存在则跳过并
#   打印本地打分命令 (把 ply_dypcd 拷回本地, PHASE=score 跑本脚本即可)。
# =============================================================================

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}
[[ -f "$PROJECT_DIR/test_dtu.py" ]] || { echo "PROJECT_DIR=$PROJECT_DIR 不是仓库根目录" >&2; exit 2; }
cd "$PROJECT_DIR"

RUN_NAME=${RUN_NAME:?需要 RUN_NAME=<log/experiments 下的 run 名>}
CKPT_KIND=${CKPT_KIND:-latest}
CKPT=${CKPT:-log/experiments/$RUN_NAME/model/${CKPT_KIND}.pth}
PHASE=${PHASE:-all}                      # all | infer | fuse | score
NUM_VIEWS=${NUM_VIEWS:-5}
RESIZE=${RESIZE:-0.8}
NUM_WORKERS=${NUM_WORKERS:-8}
FUSE_WORKERS=${FUSE_WORKERS:-8}
ROOT=${ROOT:-log/ablation/$RUN_NAME}
OUT=$ROOT/depth_cache
PLY_DIR=$ROOT/ply_dypcd
NO_SKIP=${NO_SKIP:-0}
FAST_DTU_DIR=${FAST_DTU_DIR:-$HOME/project/Fast-DTU-Evaluation}
DTU_GT_DIR=${DTU_GT_DIR:-$HOME/project/dataset/DTU/SampleSet/MVS Data}

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    set +u
    source ~/.bashrc
    conda activate uprmvs
    set -u
    export UPRMVS_MACHINE=umhpc
    export UPRMVS_PROFILE=umhpc
    PROFILE=umhpc
else
    PROFILE=${UPRMVS_PROFILE:-local}
fi
export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/models:$PROJECT_DIR/models/Depth-Anything-3/src"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
mkdir -p logs "$ROOT"

echo "======================================================================"
echo " 消融 DTU 测试  run=$RUN_NAME  phase=$PHASE  ckpt=$CKPT  job=${SLURM_JOB_ID:-manual}  host=$(hostname)"
echo " resize=$RESIZE full_image=1 views=$NUM_VIEWS  git=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo " -> $ROOT"
echo "======================================================================"

if [[ "$PHASE" != "score" ]]; then
    if [[ "$PHASE" != "fuse" ]]; then
        [[ -f "$CKPT" ]] || { echo "找不到 checkpoint: $CKPT" >&2; exit 1; }
    fi
    nvidia-smi -L || true
    args=(--phase "$PHASE" --out "$OUT" --ply-dir "$PLY_DIR" --workers "$FUSE_WORKERS")
    [[ "$NO_SKIP" == "1" ]] && args+=(--no-skip-existing)
    if [[ "$PHASE" != "fuse" ]]; then
        # 以下参数 test_dtu.py 原样转给 test_moa.py (结构从 checkpoint 的快照重建)
        args+=(--ckpt "$CKPT" --profile "$PROFILE" --split test --full-image --resize-scale "$RESIZE"
               --num-views "$NUM_VIEWS" --num-workers "$NUM_WORKERS")
    fi
    python test_dtu.py "${args[@]}"
    printf '{"run": "%s", "ckpt": "%s", "git": "%s"}\n' "$RUN_NAME" "$CKPT" \
        "$(git rev-parse HEAD 2>/dev/null || echo unknown)" > "$ROOT/test_info.json"
fi

if [[ "$PHASE" == "all" || "$PHASE" == "score" ]]; then
    if [[ -f "$FAST_DTU_DIR/eval_dtu.py" && -d "$DTU_GT_DIR" ]]; then
        echo "=== Fast-DTU-Evaluation: $PLY_DIR ==="
        PLY_ABS=$(cd "$PLY_DIR" && pwd)
        (cd "$FAST_DTU_DIR" && python eval_dtu.py --voxel_factor 1.28 --pred_dir "$PLY_ABS" \
            --gt_dir "$DTU_GT_DIR" --save --num_workers 1) 2>&1 | tee "$PLY_DIR/fast_dtu_eval.log" | grep -E "acc|overall" || true
        grep -E "^mean (acc|comp|overall)" "$PLY_DIR/fast_dtu_eval.log" || echo "WARNING 打分日志里没有 mean 行, 看 $PLY_DIR/fast_dtu_eval.log" >&2
    else
        echo "=== 跳过打分: 没有 $FAST_DTU_DIR/eval_dtu.py 或 GT $DTU_GT_DIR ==="
        echo "    把 $PLY_DIR 拷回本地同一路径后:  PHASE=score RUN_NAME=$RUN_NAME bash scripts/ablation/test_umhpc.sh"
    fi
fi
echo "=== done -> $ROOT   (汇总: python scripts/ablation/collect.py) ==="
