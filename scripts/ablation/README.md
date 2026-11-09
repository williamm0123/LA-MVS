# LAPEmvs 消融实验手册

对应论文草稿 `paper/lapemvs_manuscript/main.tex` 的表 6 (重训消融) 和表 8 (覆盖/选择)。
本文件写给几周后的自己：哪些实验、怎么投、怎么测、看哪些数、怎么下结论。

## 0. 前提 (投之前确认)

1. **FULL = 当前 `dev` 的默认配置**：LAPE + LAPE-GRU + 1/2 分辨率精修，LFR 关，`moa_gain=(1,1,1)`，
   先验训练增强关。它的 seed #1 就是已有的 `LAPE_GRU_DTU_E10` (`scripts/train_lape_gru_umhpc.sh`)。
   如果最终论文模型改了 (例如退回 E10 LAPE)，所有 B 行都要相对新的 FULL 重跑。
2. **第 0 层混合权重拿不到梯度** (`main.tex` 开头的 AUTHOR NOTES) 只存在于 E10 代码 (`0e0d057`)。
   LAPE-GRU 提交 `301dbc6` 加了先验混合 NLL (`loss.w_prior_mix`, 日志 `lape1_mixnll`)，它是第 0 层混合头
   的训练信号；2026-11-10 在真实 DTU 样本上做反传确认 `adapters.0.mixture` 有梯度。FULL 和全部消融预设都已包含。
   初始化时 RAC 常对齐失败 (`x_valid=0`)，个别样本上某一层没有梯度属正常，不是 bug。
3. 论文表 6 的 A4 数字 (0.304/0.281/0.293) 是 E10 LAPE，不是 FULL；表里要换成 FULL 的结果。
4. 所有行共用：DA3 特征 (A0FPN 除外)、单卡 A100-80GB、batch 2、lr 3e-4、warmup 1000、
   warp 128x4、multi-scale、bf16、deterministic、10 epoch (13,548 步/epoch)、同一 DTU train 列表。

## 1. 实验表

`python ablation.py --list` 打印同一张表。开关定义在 `base/config_moa.py` 的 `LAPEConfig` 第 (8) 段，
会写进 checkpoint 的结构快照，测试时自动按训练时的结构重建。

| ID | 论文行 | tier | 改了什么 | 回答什么问题 |
|---|---|---|---|---|
| FULL | A4 | 1 | 无 (额外种子 #2 #3) | 噪声量级；所有 Δ 的基准 |
| A0 | A0 | 1 | 无单目先验 (`moa.enabled=False`)，DA3 只做匹配特征，深度头不跑 | 先验总共带来多少 |
| A1 | A1 | 1 | MoA cascade：先验只决定窗口中心 (`lape.enabled=False`) | 覆盖路径单独贡献 (诊断预期 ≈0) |
| A1D | A1-D* | 1 | 简单融合：候选中心取 pdf、窗口内重归一化、无质量门控、无 v_M 门控 | 对比最接近的先例 (M2Depth 式) |
| B4 | B4 | 1 | σ 用解析值 √(s_ho²+σ_par²)，不过校准器 | C2：support-aware σ 的校准器有没有用 |
| B5 | B5 | 1 | 候选中心取 pdf 并重归一化，无 bin 积分、无质量门控 | C3b：bin 质量 + M_min 门控 |
| B6 | B6 | 1 | v_M ≡ 0，先验在匹配可靠处也生效 | C3b：可靠性门控 |
| B7 | B7 | 2 | 去掉代价适配器 (只留 logit 注入) | 两条注入路径各自贡献 |
| B8 | B8 | 2 | γ_max = 0 (只留代价适配器) | 同上 |
| B9 | B9 | 2 | 第一级只跑一遍 (先验从 stage 2 起才进入) | 两遍 stage 1 是否值得 |
| B10 | B10 | 2 | 关法向一致证据 (NCE) | NCE 的贡献 |
| NOGRU | 新 | 2 | 去掉 LAPE-GRU 专家 | GRU 的贡献 (FULL 若含 GRU 必须有) |
| NOREF | 新 | 2 | 去掉 1/2 分辨率精修 | 精修的贡献 |
| A0R | 新 | 2 | A0 + 精修 (精修不需要先验) | 拆开 "精修" 与 "先验"：FULL−A0 里有多少其实是精修 |
| A0W | A0-W | 2 | A0，基础半宽 ×1.5、候选 +50% (72/24/12/6) | 多花算力加宽窗口能否达到同样效果 |
| A0FPN | (附加) | 3 | A0 再去掉冻结 ViT (纯 FPN) | DA3 特征本身的价值，默认不投 |

\* 论文原定义 A1-D 是 "A1 + 简单融合"。A1 (MoA cascade) 没有选择路径，所以这里实现为
"FULL 的选择路径换成简单融合 (= B5 + B6)"。写论文时按这个定义描述，或改名为 "simple fusion"。

**没有重训的行**：B1/B2/B3 (RAC 结构、样本内残差) 只进校准表 9；B11 (LFR、增益上限) 已是 FULL 的默认
(关)，在附录说明即可。推理期敏感性 (表 7: M_min、γ_max、β、均匀 π) 和先验损坏实验不需要重训，
用 FULL 的 checkpoint 跑。

## 2. 投作业 (umhpc)

```bash
cd ~/project/lapemvs && git checkout dev && git pull && mkdir -p logs
DRY=1 bash scripts/ablation/submit_all.sh        # 先看会投哪些 (默认 tier 1+2 + A0/FULL 的种子 #2 #3, 共 18 个)
bash scripts/ablation/submit_all.sh              # 投
TIERS="1" bash scripts/ablation/submit_all.sh    # 只投 tier 1
EXPS="B5 B6" SEEDS=0 bash scripts/ablation/submit_all.sh   # 只投指定的
EXP=A0 SEED=2 sbatch -J abl_A0_S2 scripts/ablation/train_umhpc.sh   # 单个
```

- 每个作业 `--partition=gpu-a100 --gres=gpu:1`，脚本开头检查是 A100 且 ≥70 GiB，否则直接退出。
- run 名：`ABL_<ID>_E10`，种子 #2/#3 加 `_S2/_S3`；实际种子 = 20260526 + 1000×(编号−1)。
- 3 天时限：超时前 15 分钟存 `latest.pth` 并自动续投 (最多 8 次)。手动续训：
  `FRESH=0 EXP=<ID> SEED=<n> sbatch scripts/ablation/train_umhpc.sh`。
- 训完 (退出码 0) 自动投测试作业 (`AUTO_TEST=0` 关掉)。
- FULL 的 seed #1 不由这里训练，训完后手动测：`RUN_NAME=LAPE_GRU_DTU_E10 sbatch scripts/ablation/test_umhpc.sh`。
- 看进度：`squeue -u $USER`，`tail -f logs/abl_<ID>_<jobid>.out` (看 `eta=`)，
  tensorboard 在 `log/experiments/<run>/tensorboard`。

## 3. 测试 / 推理

`scripts/ablation/test_umhpc.sh` (一张 A100)：test_moa 推理 → test_dtu 融合 → Fast-DTU 打分。
口径与 E10 / moa1 相同：22 个测试 scan，0.8 整幅 960×1280，5 视角，bf16，conf_last>0.55 + 动态几何一致性。
默认用 `latest.pth` (退火到底；E10 的论文数字也是 latest)，`CKPT_KIND=best` 改用 best。

```bash
RUN_NAME=ABL_A0_E10 sbatch scripts/ablation/test_umhpc.sh          # 推理 + 融合 + (能打分就打分)
PHASE=fuse  RUN_NAME=ABL_A0_E10 sbatch scripts/ablation/test_umhpc.sh   # 只重融
PHASE=score RUN_NAME=ABL_A0_E10 bash scripts/ablation/test_umhpc.sh     # 只打分 (本地)
```

输出在 `log/ablation/<run>/`：`depth_cache/metrics.json`、`ply_dypcd/*.ply`、`ply_dypcd/fast_dtu_eval.log`。

**打分**需要 Fast-DTU-Evaluation 和 DTU SampleSet GT (`FAST_DTU_DIR`、`DTU_GT_DIR`，默认
`~/project/Fast-DTU-Evaluation` 和 `~/project/dataset/DTU/SampleSet/MVS Data`)。umhpc 上没有就跳过：
把 `log/ablation/<run>/ply_dypcd/` 拷回本地同一路径，`PHASE=score` 本地跑 (E10 就是在本地 5060 Ti 打的分)。

## 4. 汇总

```bash
python scripts/ablation/collect.py      # -> 屏幕 + log/ablation/summary.md (按实验, 种子取 mean±std) + summary.csv (逐 run)
```

## 5. 指标

| 列 | 含义 | 来源 | 越好 |
|---|---|---|---|
| Acc. / Comp. / Overall | DTU 点云精度 / 完整度 / 二者平均 (mm) | Fast-DTU-Evaluation | 低 |
| MAE | 最终深度绝对误差均值 (mm)，GT 有效且在深度范围内的像素 | test_moa `overall.abs_err` | 低 |
| Med. | 深度误差中位数 (mm) —— 内点精度，决定点云 | `overall.median` | 低 |
| <1/<2/<4 mm | 误差小于阈值的像素比例 | `overall.acc_*mm` | 高 |
| Cov2/3/4 | GT 落在第 s 级候选区间 (含两端半个间隔) 内的比例 —— **覆盖** | `window.cov*` | 高 |
| E_in | 最终窗口内像素的深度 MAE —— **选择** | `window.e_in_mm` | 低 |
| E_out | 最终窗口外像素的深度 MAE | `window.e_out_mm` | — |
| out px% / out err% | 窗口外像素占比 / 它们占总误差的比例 (E10：6.2% / 84%) | `window.out_*_share` | 低 |
| ΔOverall / ΔMAE | 该行 − FULL | collect.py | 负 = 更好 |

论文表 6 用 Acc/Comp/Overall/MAE/<2mm/Cov4；表 8 用 Cov2–4、E_in、E_out、out share。

## 6. 怎么读结果

- **噪声**：先看 FULL 和 A0 三个种子的 std。|Δ| 小于约 2×std 的行写 "无显著差异"，不要当作贡献。
- **先验有没有用**：FULL vs A0 (Overall、MAE)。再减去 A0R−A0 (精修本身的收益)，剩下才归功于先验。
- **覆盖 vs 选择**：A1 vs A0 的 Cov4 —— 诊断预期覆盖路径几乎不起作用。FULL vs A0 若 Cov4 相近而
  E_in 降低，说明收益来自选择路径。
- **新颖点 C3b**：B5、B6、A1D 相对 FULL 的变化。预期在先验出错的区域最明显，配合先验损坏实验 (图 robust) 一起看。
- **新颖点 C2**：B4 的深度指标 + 校准表 9 (1σ/2σ 覆盖、NLL、AUROC)。
- **点云与深度不一致**：E10 的经验是 MAE 降了而 Chamfer 持平 (被几何过滤丢掉的是粗差)。两组指标都要报，
  看 Med. 和 <1mm 判断内点精度有没有动。
- 不管结论好坏都报 (草稿的 pending 要求)。

## 7. 进度清单

- [ ] 确认 FULL 定义 / 是否先修第 0 层梯度
- [ ] `LAPE_GRU_DTU_E10` 训完并测试 (FULL seed #1)
- [ ] tier 1：A0 A1 A1D B4 B5 B6 + A0/FULL 种子 #2 #3
- [ ] tier 2：B7 B8 B9 B10 NOGRU NOREF A0R A0W
- [ ] 全部打分，`collect.py` 汇总，填表 6 / 表 8
- [ ] 推理期：表 7 敏感性、先验损坏、校准表 9
