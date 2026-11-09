#!/bin/bash
# =============================================================================
# 一次性投出全部需要重训的消融 (每个一个 sbatch 作业, 各占一张 A100-80GB)。
# 训练作业训完会自动投各自的 DTU 测试作业 (scripts/ablation/test_umhpc.sh)。
#
#   bash scripts/ablation/submit_all.sh                 # tier 1 + 2 + 种子 (默认)
#   TIERS="1" bash scripts/ablation/submit_all.sh       # 只投 tier 1
#   EXPS="A0 B5 B6" bash scripts/ablation/submit_all.sh # 只投指定的
#   SEEDS=0 bash scripts/ablation/submit_all.sh         # 不投额外种子
#   DRY=1 bash scripts/ablation/submit_all.sh           # 只打印命令
#
# 额外种子: A0 和 FULL 各跑种子 #2/#3 (FULL 的 #1 就是已有的 LAPE_GRU_DTU_E10),
# 用来估计噪声, 其余行单种子。实验表与指标见 scripts/ablation/README.md。
# =============================================================================
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
mkdir -p logs

TIERS=${TIERS:-"1 2"}
SEEDS=${SEEDS:-1}
DRY=${DRY:-0}
if [[ -z "${EXPS:-}" ]]; then
    # FULL seed 1 already exists (LAPE_GRU_DTU_E10): only its extra seeds are submitted
    EXPS=$(python3 -c 'import sys, ablation; t = {int(x) for x in sys.argv[1].split()}
print(" ".join(k for k, v in ablation.EXPERIMENTS.items() if v[1] in t and k != "FULL"))' "$TIERS")
fi

submit() {   # $1 = EXP, $2 = seed number
    local suffix=""
    [[ "$2" != "1" ]] && suffix="_S$2"
    local cmd=(sbatch -J "abl_$1$suffix" --export=ALL,EXP=$1,SEED=$2 scripts/ablation/train_umhpc.sh)
    echo "${cmd[*]}"
    [[ "$DRY" == "1" ]] || "${cmd[@]}"
}

for e in $EXPS; do
    submit "$e" 1
done
if [[ "$SEEDS" == "1" ]]; then
    for e in A0 FULL; do
        for s in 2 3; do
            submit "$e" "$s"
        done
    done
fi
echo "=== 查看: squeue -u \$USER ;  汇总: python scripts/ablation/collect.py ==="
