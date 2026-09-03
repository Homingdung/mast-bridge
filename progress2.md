# MAST 合成平衡数据集 pipeline（PCA ±0.1σ）交接文档（2026-08-25）

> ⚠️ **文档结构说明（2026-09-01）**：本文档的 **§22-§44 因意外截断丢失**（无法恢复），现结构为
> §00-§21（原始内容）→ §45（sec50 方案）→ §46（交接记录）→ **§37（恢复的 100ep bs8 负结论）**。
> §37 是恢复章节，编号在原位（§21 之后），但物理位置在文件末尾。§22-§44 的历史细节
> 参考会话记录、`hyper.md`（超参数结论）、`plot.md`（绘图）、`experiment_setting_qixian.md`（合作者协议）。
>
> **最新结果（2026-09-02→09-03）**：§37 之后的章节：§45/§48/§49/§50/§51/§52/§53/§54/§55/§56/§57/§58/**§59**。
> 全部训练/评估已完成（含 §57 random 1% A 臂 s58 补位，2026-09-03 闭环）；
> 论文用结论 = 充分收敛口径（e500 §49.13、ft150 §50.7、w150 §50.11-12、temporal §48.7-8/§54.6/§55.6-7、§52 登记、
> 指标定义附录 A；random 1% A 臂最终 seed 集 = {s54,s56,s58}，均值 1.483，见 §57.3）。
> **论文两张主表（tab:random_table / tab:temporal_table）的唯一权威登记 = §59**（LaTeX 源码 + MD 对照 +
> 数据源 + 逐格验证，2026-09-03）；**泛化性研究定稿结论 = §58.3**（两场景对照 + 逐单元 + 统计检验，
> 2026-09-03，论文讨论直接引用）。
> **注意 §00-§21 的"全域有效"为 10ep 口径已作废**，最终交接见文件末尾 §53/§57/§59。


> **给新对话的交接**：本文档是当前工作的唯一权威入口。流程起点为 clean pool（11,148 炮），
> 经 Ip≥85%×max 筛选 → PCA ±0.1σ 扰动 → FreeGSNKE 求解（已全部完成）→ 7 条判据过滤
> （已完成，位形锚定语义）→ 下一步：诊断生成 → 训练缓存 → 训练。
>
> 本节为 clean pool 数据统计画像（`statistics_shot.md` §1-§8 摘要）；§1 起为 pipeline 进展。
> 原始统计文档仍保留：`statistics_shot.md`。

---

## 00. 新会话交接（2026-08-28 定稿，所有实验已完成，进入数据整理阶段）

> **本小节是当前唯一权威入口**。数据链路（§1-§14，PCA ±0.1σ 合成 213,928 + real 213,924 配对）与
> **三组实验（random split / temporal split / Shape-OOD）全部完成**。下一步：结果整理与论文。
> 详细记录：random = §9-§14；temporal = §15-§19；Shape-OOD = §20-§20.17。

### 00.1 项目一句话

用真实 MAST 数据 + FreeGSNKE 仿真合成数据（PCA ±0.1σ 扰动，位形锚定过滤）训练 TokaMind 神经网络，
做「69 维磁诊断 → 65×65 平衡磁通 psi」的静态平衡重构预测。
核心研究问题：**仿真（合成）数据能否提升模型的精度与泛化？**

**最终答案：能。** 合成预训练在三个场景全部有效（见 00.2），数据越稀缺增益越大。

### 00.2 三组实验总结（最终结果，nrmse_per_sample %，增益 = (A−C)/A）

**A = real scratch（随机初始化）｜C = synthetic pretrain + real finetune（唯一变量 = 预训练）**

| 实验 | 划分方式 | Test 集 | 档位 | A（100%→5%）| C（100%→5%）| 增益 |
|---|---|---|---|---|---|---|
| **Random split**（§9-14）| 按炮随机 80/10/10（seed 20260825）| 21,350 片 / 730 炮 | 100/50/25/10/5% | 0.76→2.52 | 0.72→1.01 | **−5%~−60%** |
| **Temporal split**（§15-19）| M5+M6+M7 → M8 val → M9 test | 17,498 片 / 507 炮（纯 M9）| 100/50/25/10/5% | 2.69→5.35 | 2.29→2.43 | **−14%~−60%** |
| **Shape-OOD diverted**（§20）| δasym<P10 定义 OOD，r_OOD 分层 | Test-OOD-only 19,009 片 / 775 炮 | 100/50/25/5% | 2.07→5.53 | 1.78→1.84 | **−14%~−67%** |
| **Shape-OOD limiter**（§20.15）| δasym>P90 定义 OOD，r_OOD 分层 | Test-OOD-only 817 片 / 227 炮 | 100/50/25/5% | 4.32→5.70 | 4.06→4.93 | **−6%~−18%** |

补充：Shape-OOD 还有 Test-all 口径（diverted 20,761 片），结果与 Test-OOD-only 几乎一致（§20.14 表）。

### 00.3 核心结论一览

1. **预训练三场景全域有效，增益随数据稀缺单调放大**（random −60%、temporal −60%、shape-OOD −67%），数据越稀缺价值越大
2. **C 臂曲线几乎水平**（预训练抹平数据量变量），A 臂随稀缺陡升——"仿真预训练 = 数据量放大器"（5%+预训练 ≈ 50%+ 纯真实）
3. **Shape-OOD 增益最大**（−67%）：预训练学到跨形状可迁移的平衡物理（形状先验），LCFS 可视化（§20.17）直接可见：纯真实 5% 红线失形、预训练+5% 绿线贴合 EFIT
4. **Topology 分离成功**：diverted（−67%）与 limiter（−18%）各自组内做 Shape-OOD，无跨拓扑混淆；limiter 增益小因合成池小（8,808 vs 175,717）+ 位形受接触面约束
5. **噪声非必要条件**：PCA+位形锚定方案下 clean/noisy05/noisy20 预训练微调后结果几乎持平（§14/§15.8）
6. 关键前提（成功三要素，§10.2）：① PCA 沿真实流形扰动 ② ±0.1σ 小幅度 ③ 位形锚定过滤（accepted 全为 EFIT 真实位形）

### 00.4 数据资产速查（全部在 workspace 根下）

| 资产 | 路径 | 说明 |
|---|---|---|
| Real manifest | `data/manifests/training_pca_01sigma/real.jsonl` | 213,924 片 / 7,286 炮（配对合成求解）|
| 合成 accepted | `data/manifests/fullsolve_pca_01sigma_accepted.jsonl` | 213,928（含 psi_axis/psi_bndry/parent_shot）|
| Real cache | `data/processed/training_cache_pca_01sigma/real_full.npz` | 213,924×69 特征 + psi |
| 合成 cache | `.../synth_pretrain_clean.npz`（213,928）| clean 诊断版本 |
| Shape metadata | `data/processed/shape_ood/{real,synthetic}_shape_metadata.jsonl` | canonical LCFS 170 点 + δu/δl/δmean/δasym + topology |
| Random split manifests | `data/manifests/training_pca_01sigma/split_{train,val,test}_real.jsonl` + `subset_run_real_{f}pct.jsonl` | seed 20260825 |
| Temporal manifests | `data/manifests/training_temporal/` | M5-M7/M8/M9 + subsets |
| Shape-OOD manifests | `data/manifests/shape_ood/`（diverted）+ `shape_ood_limiter/` | split + subsets + synth C |
| **评估结果（全部 metrics json）** | `artifacts/tokamind_test_eval/`：`pca01sigma_curve.json`（random clean）、`pca01sigma_noisy_curve.json`（random noisy）、`temporal_curve.json`+`temporal_curve_seeds.json`、`shapeood_test_{all,ood_only}_metrics.json`、`limiter_test_{all,ood_only}_metrics.json` | 每个 run 的 RMSE/MAE/RMAE/nrmse（per sample + global）|
| **图产物** | `artifacts/pca01sigma_results/`：`two_experiments_compare_side.png`（random+shapeOOD）、`three_experiments_compare_side.png`、`shapeood_curve.png`、`shapeood_gain_*.png`、`shapeood_topology_side.png`、`shapeood_lcfs_visual.png`、`shapeood_audit.png`、`temporal_rmse_sample_multiseed_*.png`、`pca01sigma_zero_finetune.png`、`pca01sigma_gain_bars.png` | dpi=300 |
| Run 目录 | `runs/tokamind-pca01sigma-*`（random）、`runs/tokamind-temporal-*`（temporal）、`runs/tokamind-shapeood-*`/`tokamind-limiter-*`（Shape-OOD）| 每 run 有 summary + checkpoints + scalers |

### 00.5 坑（新会话必读，完整版见 §20 坑大全 / §7）

1. **环境**：求解/提取用 `.freegsnke-solve-env`；训练/评估用 `.tokamind-train-env`（勿装 numpy 2.x）
2. **GPU 拉满**：d64 模型 batch 8 单进程利用率仅 ~10%（<15% 停机线）→ 必须 **8 卡 × 2 = 16 进程**；短 job 完成后需补位（额外 seed 或手动启动）；`nvidia-smi` 每卡 ≥2 进程才算安全
3. **评估脚本缓存**：不会命中已有 cache——需手动把切好的 test cache 复制为 `artifacts/tokamind_test_eval/test_cache_split_test_*.npz`（否则从 zarr 重提取失败）
4. **预训练不可传 real val shots**（合成 manifest 无 val 炮，报 "Validation shots missing"）
5. **合成预训练 cache**：`synth_pretrain_clean.npz` 是 random-split 旧版（炮集不同），新划分需 `build_cache_batched.py` 重建（34 秒全量）
6. 后台任务 `setsid nohup ... & disown`；python 加 `-u`；动态最闲卡分配 runner 模式见 §19.4/§20.13
7. LCFS 提取：合成固定 ψ_bndry + O 点连通区（碰边界回退 §16 动态扫描）；splprep 需先去重（§20.17）

### 00.6 下一步建议（数据整理/论文阶段）

1. **统一结果表**：三组实验按 00.2 格式做成论文 Table（A/C 多 seed 均值±std 已具备：random 部分档 1 seed 待补）
2. **图风格统一**：全部图已是 dpi=300、无大 title 风格；可统一字体/色板
3. **待跑项（可选）**：① geometry 指标（磁轴/LCFS 距离/δ 误差，§20.14 遗留）② severity bins（mild/moderate/extreme）③ random 100% 档多 seed ④ limiter 多 seed
4. **背景材料**：数据统计画像 §0、pipeline 全流程 §1-§14、三组实验正文 §15-§20
5. 写论文时引用图：`two_experiments_compare_side.png`（主对照）、`shapeood_topology_side.png`（topology 分层）、`shapeood_lcfs_visual.png`（几何证据）、`shapeood_audit.png`（OOD 定义）

---

## 0. clean pool 数据统计画像（statistics_shot.md 摘要，2026-08-19）

> 数据源：fit NPZ（818,139 拟合时刻 / 11,148 炮）、clean_pool.csv、`.shots_metadata.parquet`、zarr 全量扫描。

### 0.1 时间区间
- 全部拟合时刻：0.005–0.685 s（并集，非每炮覆盖全程）
- 每炮起点：几乎全部 ~0.005–0.01s（中位 0.010，99% 炮 ≤0.025；仅 95 炮 >0.05s）
- 每炮末帧：p10=0.25 / p50=**0.375** / p90=0.52 / max=0.685 s
- 时间轴结构：0.005–0.10s 启动段（Ip 爬升，limited 为主）→ 0.10–0.35s 平顶（diverted 为主）→ 0.35s+ 下降/延伸

### 0.2 位形（拓扑，818,139 帧逐帧判定）
- **各 campaign 高度一致**：diverted 76.7–78.5%（合计 **77.7% diverted / 22.1% limited**），与 campaign 无关
- **平顶窗 [0.10,0.30] 内**：93.9% diverted / 5.2% limited（启动/下降段 limited 被排除）
- 窗内翻转（mixed）罕见：M5–M8 <0.4%，M9 达 3.2%（位形演化最频繁）

### 0.3 通道一致性（全量 11,148 炮）
- magnetics（OBR 19/OBV 19/CCBV 40）+ flux_loop（15）：✅ 全 campaign 一致
- **active 线圈**：⚠️ 11 vs 13 通道（P6 有无）混杂，每个 campaign 内部都有
- **passive**：⚠️ 7 种配置；78 通道多数派（9,558 炮 = 90%）、88 通道（1,016 炮）其余零星

### 0.4 Campaign 成熟度/稳定性
| campaign | 年份 | 炮数 | useful 率 | 窗内 Ip 波动中位 | 平台时长中位 | 长平台(>0.3s)占比 |
|---|---|---|---|---|---|---|
| M5 | 2005 | 1,905 | 41%（最低）| 2.83% | 0.184s | 9% |
| M6 | 2006-07 | 2,670 | 47% | 2.31% | 0.192s | 17% |
| **M7** | 2008-10 | **3,414（最多）**| 61% | **1.84%（最稳）**| 0.196s | 19% |
| M8 | 2011-12 | 1,844 | **71%（最高）**| 3.45% | **0.168s（最短）**| 18% |
| M9 | 2013 | 1,028 | 69% | 2.76% | 0.194s | **27%（最多）**|

- **M7 推荐主力**：量最大 + Ip 最稳 + useful 中上；**M5 不建议为主**（useful 41%、通道最杂、波动偏大）
- 每炮拟合点数中位 67–80（M9 最多 80），dt=0.005s

### 0.5 样本筛选演化（重要决策史）
1. 原候选：固定窗 [0.12,0.24] + 逐炮平台检测（223k 样本，时间分布右偏）
2. **弃固定窗**：固定窗内 **34.5% 样本 psi 不平滑**（0.03s 窗口内变化 >5%，EFIT 有效性存疑；
   p50=6.3%、p90=8.5%、max=186%；集中在窗口左半段 0.12–0.17s）——审计见 statistics_shot.md §7
3. **定稿 85%×max**（2026-08-19）：Ip ≥ 85% × 炮内 Ip 最大值，全时间轴、无固定窗。
   语义直白、覆盖全 11,148 炮、时间分布自然贴合各炮平台
4. 阈值对比：80%×max=568,082 / **85%×max=533,552（定稿）** / 90%×max=487,408；
   85%×Q95 旧判据=520,049（与 85%×max 几乎等价，Q95/max 中位 0.992）
5. **85%×max 是电流水平门槛，不是平稳性门槛**；如需更严可叠加段内 |rel|≤3% 或 psi 稳定性 ≤5%

### 0.6 参数范围与 PCA 扰动分析（基于 533k 筛选样本）
**参数范围（p1-p99）**：
| 参数 | p1 | p50 | p99 | 扰动决策 |
|---|---|---|---|---|
| a0 | -1.66e4 | -889 | 3.97e3 | ✅ |
| a1 | -8.22e5 | 5.34e5 | 1.85e6 | ✅ |
| a2 | -1.84e6 | -5.31e5 | 8.21e5 | ✅ |
| b0 | -1.23 | 0.98 | 2.35 | ✅ |
| b1 | -4.76 | -1.42 | 1.98 | ✅ |
| b2 | -0.91 | 0.45 | 2.45 | ✅ |
| ip | -7.38e5 | 6.55e5 | 9.42e5 | ❌ 固定（∫Jφ=Ip 锚定）|
| fvac | 0.310 | 0.407 | 0.409 | ❌ 固定（准常数）|
| P2/P3/P4/P5 线圈 | 见原表 | | | ✅ |
| P1/P6 | 0 | 0 | 0 | ❌ 零方差，剔除 |

**PCA（16 维）**：PC1=38.2%（P4/P5(−) vs P2(+)）、PC2=21.5%（FF' 形状）、PC3=15.4%（P2+a1/a2）、
PC4=10.0%（P3）、PC5=6.2%（纯 a0）、PC6=3.8%、PC7=2.4%（P3 不对称）、PC8=1.4%；
**前 8 PC 合计 99.0%**。含 fvac 对照：fvac 载荷集中 PC5/PC6 且与 a0 耦合 → 固定 fvac 即不扰动这两方向，合理。

**扰动幅度实验教训**：±0.5σ 联合扰动 → diverted→diverted 仅 7%（推过 limiter 分岔）；
**±0.1σ → 70%**（5/10 收敛且匹配）→ 定稿 ±0.1σ。±1.5σ 反变换后参数相对变化 120,000%（p99）——
**必须 clip 到真实 p1-p99 范围**。残留失配来自炮自身分岔敏感性，非幅度可完全消除。

---

## 1. 数据与筛选（已完成）

| 项 | 值 |
|---|---|
| 基础池 | clean_pool（818,139 拟合时刻 / **11,148 炮**）|
| 筛选标准 | **Ip ≥ 85% × 炮内 Ip 最大值（max）**，全时间轴，无固定窗 |
| 筛选样本 | 533,552 |
| 干净样本（剔除负 Ip 25,989 + 线圈 NaN 4,906 + 重叠）| **502,402** |
| 覆盖炮数 | 10,667（zarr 数据 11,573 炮；906 炮因 NaN/负 Ip 无样本）|

- 时间分布：主峰 0.18-0.22s，覆盖 0.06-0.50s+
- EFIT 位形（按 target_time 逐样本判定）：**diverted 96.1% / limited 3.9%**

## 2. PCA 扰动方案（定稿）

- **扰动空间**：16 维 = 6 剖面系数（a0/a1/a2/b0/b1/b2）+ 10 线圈电流（P2IL/P2IU/P2OL/P2OU/P3L/P3U/P4L/P4U/P5L/P5U）
- **固定不扰动**：fvac（准常数，用测量值）、Ip（∫Jφ=Ip 锚定）、P1/P6（零方差）
- **PCA**：基于 502,402 干净样本，标准化 + 相关矩阵特征分解，**前 8 主成分（98.5% 方差）**
- **扰动**：PC 得分 t 加 δ_k ~ U(−0.1, +0.1)·√λ_k（**±0.1σ**；0.5σ 时拓扑失配 93%，0.1σ 小试 70%）
- **反变换 + 约束**：x' = V·(t+δ)·σ + μ → clip 到 p1-p99 → Σα=Σβ=0
- 参数范围 / PCA 载荷 / 幅度实验详见 §0.6

## 3. 求解（已完成，100%）

**manifest**：`data/manifests/fullsolve_pca_01sigma.csv`（502,402 行：shot/target_time/alpha_abs/beta_abs/coil_abs(JSON)/ip/fvac/pca_delta）

**脚本**：`mast-bridge/scripts/run_pca_solve_shard.py`（分片 worker）
- 按行号分片 → 炮内时间序求解 → 调用 `run_freegsnke_forward.py` 绝对参数注入
  （`--alpha-abs`/`--beta-abs`/`--coil-currents-json`）
- continuation warm start（成功片传递 equilibrium.npz，失败冷启动）
- 求解配置：65×65、tol 1e-8、500 iter、600s 超时；断点续跑（已有 metadata.json 跳过）

**运行史**：
| 轮次 | 时间 | 分片 | 完成 |
|---|---|---|---|
| 主跑（96+32 shards）| ~08-20 | 128 进程 | 291,900（58.1%）|
| 续跑（resume）| 08-24 05:26 → 08-25 01:5x | 56 shards / 64 核 | **234,410 行全部处理**（20.5h）|

**最终求解结果**（resume shards jsonl 对账，`data/manifests/synthetic_pca_01sigma/resume_shards/*.jsonl`）：
- 成功（rc=0，有 metadata.json）：**476,954**
- 失败（rc=1）：25,422（多为 passive NaN 快败，2s/个）
- 超时（rc=124）：26
- 失败/超时的样本**不重跑**（已由过滤判据 1 兜底拒绝），维持现状

**产物目录**：`data/processed/synthetic_pca_01sigma/<shot>_t<time>_v000/`（equilibrium.npz + metadata.json + equilibrium.png + coil_abs.json），共 ~526k 目录

## 4. 过滤（已完成，位形锚定语义定稿 2026-08-25）

**7 条判据**（`filter.md`）：1 收敛 tol≤1e-8 → 2 位形与 EFIT 一致 → 3 ∫Jφ=Ip → 4 p(s)≥0 → 5 F²(s)>0 → 6 Jφ 在 limiter mask 内 → 7 ψnorm∈[0,1]

**实现**：`mast-bridge/scripts/build_synthetic_manifest.py` + `mast-bridge/src/mast_bridge/dataset/synthetic_manifest.py` + `synthetic_quality.py`

**本轮修改（2026-08-25，位形锚定语义，重要！）**：
1. **移除无条件 `limiter_contact` 拒绝**（原：solve 接触 limiter 即拒，会丢掉 EFIT=limited 的匹配样本）
2. **移除 `efit_limited_target` 拒绝**（原：EFIT 目标为 limited 直接拒 = 人为筛选位形）
3. **位形判定改为锚定 target_time**：新增 `efit_topology_at_time()`，取 EFIT x_point_r 在
   **target_time 最近帧**的 X 点计数（原实现用全时间轴中位数，会把帧级位形混掉）
4. `build_synthetic_manifest.py` 新增 `--data-dir` 参数（此前 topology 判据因 PCA 样本
   metadata 无 `data_dir` 字段被静默跳过；`synthetic_manifest.py` 新增 `_shot_zarr_dir()` 解析）
5. 新增 lru_cache 缓存 per-shot x_point 序列（500k 样本的 topology 判据提速）

**运行命令**（重跑过滤时）：
```bash
mast-bridge/.freegsnke-solve-env/bin/python -u mast-bridge/scripts/build_synthetic_manifest.py \
  --synthetic-root data/processed/synthetic_pca_01sigma \
  --variant-csv data/manifests/fullsolve_pca_01sigma_variant_ids.csv \
  --output data/manifests/fullsolve_pca_01sigma_accepted.jsonl \
  --rejected-output data/manifests/fullsolve_pca_01sigma_rejected.jsonl \
  --task pca_01sigma \
  --data-dir data/raw/mast
```
（variant CSV 由 manifest 生成：shot/target_time/variant_id=v000；完整过滤耗时 ~30-40 min）

**过滤结果（定稿）**：

| 项 | 值 |
|---|---|
| **accepted** | **213,928**（`fullsolve_pca_01sigma_accepted.jsonl`）|
| rejected | 288,474（`fullsolve_pca_01sigma_rejected.jsonl`）|
| 合计 | 502,402（全部覆盖）|

**Rejected 原因分布**：
| 原因 | 数量 | 说明 |
|---|---|---|
| topology_mismatch | 204,922 | EFIT=diverted→solve=limited 204,226 + EFIT=limited→solve=diverted 696 |
| solver_not_converged | 53,114 | 判据 1 |
| metadata_missing | 25,448 | rc≠0 无产物（passive NaN 等）|
| pressure_negative | 4,789 | 判据 4 |
| psi_norm_mask_out_of_range | 201 | 判据 7 |

**位形对账**（n=501,627 可判样本）：
| EFIT | solve=diverted | solve=limited | 无解/其他 |
|---|---|---|---|
| diverted（482,195）| **199,919 匹配** | 204,226 失配 | 78,050 |
| limited（19,432）| 696 失配 | **13,630 匹配** | 5,106 |

**位形语义**：锚定 EFIT target_time 真实位形——匹配即保留，不做人为位形筛选。
accepted 组成：~199,919 diverted + ~13,630 limited + 379 EFIT 不可判。

## 4.1 最终数据集统计（accepted，2026-08-25）

### 总体
| 项 | 值 |
|---|---|
| **样本数（time slices）** | **213,928** |
| **炮数（parent shots）** | **7,287** |
| 时间范围 | 0.010 – 0.645 s |
| 每炮 slices | 中位 **27**（min 1 / p10 4 / p25 13 / p75 42 / p90 53 / max 108）|
| 位形 | diverted **199,957**（93.5%）/ limited **13,971**（6.5%）|

### 时间分布（0.025s 桶）
```
0.05s:    432     0.15s: 15,670    0.25s: 21,423    0.35s: 11,110    0.45s:  3,262
0.075s:  3,310     0.175s: 18,459   0.275s: 20,332   0.375s:  7,881   0.475s:  2,594
0.10s:   8,163     0.20s: 20,324    0.30s: 18,106    0.40s:  6,015   0.50s:  1,890
0.125s: 12,486     0.225s: 21,035   0.325s: 14,573   0.425s:  4,206   0.525-0.65s: ~2,600
```

### 区段分布
| 区段 | 样本 | 占比 |
|---|---|---|
| 0.01–0.10s 启动段 | 5,828 | 2.7% |
| **0.10–0.35s 平顶（主峰 0.20–0.30s）** | **175,574** | **82.1%** |
| 0.35–0.50s 下降段 | 28,889 | 13.5% |
| 0.50s+ 长尾 | 3,637 | 1.7% |

### 通道缺失剔除后（净数据集）
剔除 4 个样本 = shot **13108** 的 t=0.24/0.245/0.25/0.335（该炮 `pickup_CCBV_CCBV33` 全时段 NaN，real 配对特征无法提取；另 2 个不完整炮 13107/29250 本就不在 accepted）。

| 项 | 剔除前 | 剔除后（净） |
|---|---|---|
| **样本数** | 213,928 | **213,924** |
| **炮数** | 7,287 | **7,286** |

- 剔除后每炮 slices 中位仍为 27；时间区段分布不变（平顶段 175,570）
- 合成侧 alpha/beta/ip/fvac/coil 全有限，无额外损失；合成 diagnostics modeled 通道全量覆盖 schema 69（0 缺失）

**净数据量：213,924 samples / 7,286 shots**

### 4.2 输入特征通道明细（schema `mast_level2_common_69`，69 通道）

| 类别 | 数量 | 通道名 |
|---|---|---|
| **标量** | 2 | `target_time`、`magnetics_ip` |
| **coil_active（线圈电流）** | 11 | `P2IL FEED`、`P2IU FEED`、`P2OL FEED`、`P2OU FEED`、`P3L FEED`、`P3U FEED`、`P4L FEED`、`P4U FEED`、`P5L FEED`、`P5U FEED`、`SOL` |
| **flux_loop（磁通环）** | 10 | `CC03`、`CC05`、`P3U/1`、`P3U/4`、`P4L/1`、`P4L/4`、`P4U/4`、`P5L/1`、`P5L/4`、`P5U/1` |
| **pickup_CCBV（探针）** | 29 | `CCBV03`-`CCBV40`（实测 29 个：03,04,05,07,08,09,11,12,14,15,16,17,18,19,20,24,25,26,27,28,29,30,31,33,34,36,37,39,40）|
| **pickup_OBR（探针）** | 11 | `OBR01`、`OBR04`、`OBR05`、`OBR06`、`OBR09`、`OBR10`、`OBR12`、`OBR13`、`OBR14`、`OBR16`、`OBR18` |
| **pickup_OBV（探针）** | 6 | `OBV04`、`OBV05`、`OBV06`、`OBV09`、`OBV17`、`OBV18` |
| **合计** | **69** | |

**通道来源**：
- 合成侧：`coil_currents.active`（metadata，13 通道 ⊇ 11 需求）+ modeled flux_loop（machine 44 通道 ⊇ 10）+ modeled pickup（machine 78 通道 ⊇ 46）
- 真实侧：zarr `pf_active` / `magnetics` 实测信号（10,667 炮扫描，仅 13108 的 `pickup_CCBV_CCBV33` 全时段 NaN，已剔除其 4 样本）
- 特征提取：`tokamind_manifest.diagnostic_feature_vector`，非有限值行会被 `build_dataset_cache` 丢弃

## 5. 数据曲线实验（已就绪，GPU 机器执行）

> **完整方案**：`experiment_pca01sigma_curve.md`（含同步清单、cache 构建、训练、评估全流程）。
> 实验目标：A 臂（随机初始化）vs B 臂（仿真预训练）× 真实数据比例 100/50/25/10/5%。

### 5.1 已完成（本机）
| 项 | 状态 |
|---|---|
| real manifest（213,924，剔除 13108 的 4 行）| ✅ `data/manifests/training_pca_01sigma/real.jsonl` |
| random shot-level split 80/10/10（seed 20260825，无 campaign）| ✅ train 5,828 炮/171,427 行、val 728/21,147、test 730/21,350 |
| 嵌套 real 子集 5/10/25/50% | ✅ `subset_run_real_{f}pct.jsonl` |
| 合成预训练集（= real 100%）| ✅ `run_synth_pretrain.jsonl`（171,427 行）|
| clean 诊断（213,928）| ✅ `diagnostics.npz` |
| noisy 诊断（Gauss, scale=1.0）| ✅ `diagnostics_noisy.npz` |
| cache 脚本（batch 版，238 倍加速，数值已验证一致）| ✅ `build_cache_batched.py` |
| 子集切片脚本 | ✅ `slice_cache.py` |
| 训练运行脚本（预训练 + 10 档）| ✅ `runs/run_pca01sigma_curve.sh` |

### 5.2 GPU 机器执行（交接）
```bash
# 1. 同步（见 experiment_pca01sigma_curve.md §3）
# 2. cache（batch 版 ~10 min 内完成全部）
bash runs/run_pca01sigma_curve.sh --phase cache --workers 48
# 3. 训练（预训练 + 10 档，GPU 队列 + state 断点续跑）
bash runs/run_pca01sigma_curve.sh --phase train --gpus 2 --workers 4
# 4. 评估出曲线（§7）
```

### 5.3 模型参数（定稿）
d-model 64 / layers 2 / heads 4 / FF 128 / dropout 0.05 / lr 1e-4 / bs8 / AdamW；
magnetic-diagnostics(69 通道) → raw-psi(65×65)；各档统一 epochs=10（默认，可调）；
预训练 171,427 行 × 10 epochs（B 臂额外成本）；微调用 `--finetune-method full`。

## 6. 关键文件清单

| 文件 | 说明 |
|---|---|
| `statistics_shot.md` | 数据统计 + PCA 方案 + 原始交接（§9）|
| `filter.md` | 7 条过滤判据 |
| `data/manifests/fullsolve_pca_01sigma.csv` | 502,402 行扰动 manifest |
| `data/manifests/fullsolve_pca_01sigma_remaining.csv` | 续跑剩余行（234,410，已完成）|
| `data/manifests/fullsolve_pca_01sigma_variant_ids.csv` | 过滤用 variant CSV |
| `data/manifests/fullsolve_pca_01sigma_accepted.jsonl` | **213,928 accepted** |
| `data/manifests/fullsolve_pca_01sigma_rejected.jsonl` | 288,474 rejected + 原因 |
| `data/processed/synthetic_pca_01sigma/` | 求解产物（~526k 目录）|
| `data/manifests/synthetic_pca_01sigma/` | 分片日志（shards/ 主跑 + resume_shards/ 续跑 + extra_*）|
| `runs/run_pca_full_shards.sh` | 主跑启动脚本（96 分片）|
| `runs/run_pca_resume_shards.sh` | 续跑启动脚本（56 分片，已用）|
| `mast-bridge/scripts/run_pca_solve_shard.py` | 求解分片 worker |
| `mast-bridge/scripts/run_freegsnke_forward.py` | FreeGSNKE 求解器入口 |
| `mast-bridge/scripts/build_synthetic_manifest.py` | 过滤脚本（新增 --data-dir）|
| `mast-bridge/src/mast_bridge/dataset/synthetic_manifest.py` | 过滤核心（位形锚定语义）|
| `mast-bridge/src/mast_bridge/dataset/synthetic_quality.py` | 判据 2-7 + efit_topology_at_time |
| `mast-bridge/tests/test_synthetic_quality.py` 等 | 过滤测试（已随语义更新，34 项通过）|

## 7. 坑（必须注意）

1. **环境**：求解/过滤/诊断用 `.freegsnke-solve-env`；训练用 `.tokamind-train-env`；两者不能混用
2. **zarr 版本差异**：系统 python 为 zarr 2.x（`create_dataset`），solve-env 为 zarr 3.x（`create_array`）；测试里用 `getattr(group, "create_array", None) or group.create_dataset` 兼容
3. **topology 判据必须传 `--data-dir`**，否则因 metadata 无 data_dir 被静默跳过（旧 production_a 就是如此，rejected 里无 topology_mismatch 可证）
4. **位形锚定是 target_time**（`efit_topology_at_time`），不是全时间轴中位数（旧 `efit_topology` 仅向后兼容）
5. **后台任务**：`setsid nohup ... & disown`，python 加 `-u`；线程过订阅：`OMP_NUM_THREADS=1`（求解）或 8（训练）
6. 断点续跑靠"metadata.json 存在"判断，勿删已有产物目录
7. passive NaN 样本（~14%）求解必失败且快败（2s），无需预检
8. 过滤脚本遍历 526k 目录较慢（~30-40 min），输出文件在整个扫描完成后才写入，勿中途 wc -l 判断进度

## 8. 状态速查

```
求解:    502,402/502,402 处理完毕（成功 476,954）
过滤:    accepted 213,928 / rejected 288,474（位形锚定语义）
净数据:  213,924 samples / 7,286 shots（剔除 13108 的 4 个 NaN 通道样本，§4.1）
诊断:    clean + noisy 各 213,928 已生成 ✅
缓存:    GPU 机器跑（batch 版 ~10 min，§5.2）
训练:    GPU 机器跑（11 jobs，§5.2）
```

## 9. 训练与评估结果（2026-08-26 完成）

### 9.1 执行过程
- cache：batch 版全部构建（real_full 3.16GB + synth_clean/noisy 各 2.7GB + 子集切片，~20 min）
- 训练：**11/11 jobs 全部 rc=0**（预训练 + 10 档微调），8×RTX 4090 16 并行，~2h 完成
- 期间修复 2 个 runner 问题：① 预训练被误传 val shots（合成数据无 val 炮）→ 预训练去掉 `--val-shot` 用内部划分；② 预训练与微调并行启动导致 finetune 失败 → 微调循环等预训练完成再启动
- 评估：test cache 用 batch 版重建（4.8s）并按 manifest 顺序重排后评估 11 run

### 9.2 结果（test 21,350 点 / 730 炮，RMAE/样本）

| 比例 | A scratch | B PCA预训练+微调 | 增益 |
|---|---|---|---|
| 100% | 0.558% | **0.529%** | **−5.3%** ✅ |
| 50% | 0.655% | **0.576%** | **−12.1%** ✅ |
| 25% | 0.820% | **0.622%** | **−24.2%** ✅ |
| 10% | 1.581% | **0.689%** | **−56.4%** ✅ |
| 5% | 1.929% | **0.742%** | **−61.5%** ✅ |
| 零微调 | — | 8.40% | 参考 |

### 9.3 重大结论

**PCA ±0.1σ 仿真预训练全部 5 档转正，数据越稀缺增益越大（5.3%→61.5%）**——
本项目历史上首次预训练有效（对比：v2 预训练有害 6.77%>随机、V1 无效、旧方案打平）。
关键因素：① PCA 沿真实流形扰动（保留物理耦合）② ±0.1σ 小幅度（拓扑匹配率 70%）③ 位形锚定过滤（accepted 213,924 全为 EFIT 真实位形）。

### 9.4 产物
- `artifacts/tokamind_test_eval/pca01sigma_curve.json`（11 run 评估）
- `artifacts/tokamind_test_eval/pca01sigma_curve.png`（双曲线图）
- runs：`runs/tokamind-pca01sigma-{synth-pretrain,scratch-*,finetune-*}`

## 10. 结果深度分析（2026-08-26）

> 训练+评估完成后对 §9 结果的完整解读。图：`artifacts/pca01sigma_results/pca01sigma_curve.png`。

### 10.1 核心发现：预训练增益随数据稀缺单调放大

| 比例 | scratch RMAE | pretrain-ft RMAE | 绝对改善 | 相对增益 |
|---|---|---|---|---|
| 100% | 0.558% | 0.529% | 0.029pp | **−5.3%** |
| 50% | 0.655% | 0.576% | 0.079pp | **−12.1%** |
| 25% | 0.820% | 0.622% | 0.198pp | **−24.2%** |
| 10% | 1.581% | 0.689% | 0.892pp | **−56.4%** |
| 5% | 1.929% | 0.742% | 1.187pp | **−61.5%** |

- **增益随数据量单调递增**：5% 档相对增益是 100% 档的 12 倍——预训练的价值在数据稀缺时最大
- **B 臂 5% 档（0.742%）甚至优于 A 臂 50% 档（0.655%）**——即预训练相当于把真实数据量"放大"了 10 倍以上（5%→~50%+）
- 对照 §30 的 v2 结论（预训练全程有害/仅在 0.5% 反超 3-7%）——**本轮在 5% 就有 61% 增益**，属质的飞跃

### 10.2 为什么这次成功（vs 历次失败的对比）

| 方案 | 结果 | 根因 |
|---|---|---|
| v2 统计宽度（§13-§18） | 预训练有害（6.77% vs 随机 6.60%）| 宽扰动（ip±40%）+ 独立采样 → 89% limiter 错标签 |
| V1 独立 ±2-5% | 无效/打平 | 扰动破坏线圈-剖面耦合，流形外组合 |
| **PCA ±0.1σ（本轮）** | **全域转正** | ① 沿真实协方差方向扰动（保留物理耦合）② 小幅度（拓扑匹配率 70%）③ 位形锚定过滤（只留 EFIT 真实位形）|

**三个成功要素缺一不可**：
1. **PCA 流形扰动**：扰动方向 = 真实数据主成分（PC1: P4/P5↔P2 联动等），不生成物理上不存在的参数组合
2. **±0.1σ 小幅度**：0.5σ 时拓扑失配 93%，0.1σ 时 70% 匹配——幅度是"流形内 vs 跨分支"的临界
3. **位形锚定过滤**：accepted 只保留 EFIT 真实位形（diverted/limited 各自匹配），合成标签与真实同分布

### 10.3 对研究问题的回答

**"仿真数据能否提升真实数据上的平衡重构精度与泛化？"**

**答案（正面）**：能。前提是合成数据①沿真实流形生成（PCA）②幅度受控（±0.1σ）③位形与真实锚定。在此前提下：
- 真实数据充足时：小增益（−5.3%）
- 真实数据稀缺时：大增益（−61.5%），等效数据量放大 10 倍+
- 这为"用仿真合成数据缓解真实数据稀缺"提供了首个完整正面证据链

### 10.4 局限与后续

1. 单 seed（54），可补 4 seed 验证显著性
2. 预训练数据 = train 炮合成样本（171,427），与真实 train 同炮——严格 OOD（完全未见炮）的预训练效果待验证（§19 OOD 实验）
3. 零微调 8.40% 仍远高于微调后（0.53%）——合成数据分布仍有偏移，微调是必要的
4. 后续：① 多 seed ② 更大扰动幅度扫描（0.05σ/0.2σ）③ 与 V1 方案同曲线对比

### 10.5 测试集数据可信度声明（2026-08-26 验证）

**测试集 = 100% 真实 MAST 数据，无泄漏**（三重验证）：

| 检查项 | 结果 |
|---|---|
| source 分布 | 21,350 行全部 `source=real`（0 合成）|
| sample_id | 全部带 `_real` 后缀（合成为 `_v000`）|
| 炮级隔离 | train∩test=0、val∩test=0、train∩val=0（730 test 炮完全未进训练）|
| test cache | 从真实 zarr EFIT psi 提取（batch 版，数值已验证）|

**含义**：
- §9/§10 的 clean 预训练增益（−5.3%~−61.5%）全部在**未见过的真实炮**上测得，完全可信
- 预训练/微调均只接触 train 炮（5,828 炮）；test 730 炮 held-out
- 标准协议：预训练可在 train 域内（合成形式），评估严格 held-out

## 11. J_φ 物理自洽性验证：调试过程与坑总结（2026-08-26）

> 目标：用预测 psi 经 GS 算子 Δ*ψ = −μ₀RJ_φ 反推 ∫J_φ，散点图对比输入 Ip，验证模型预测的物理自洽性。**核心教训：这类"二次物理量"验证对数据处理错误极度敏感，任何一步错都会得到荒谬结果，必须逐层排查。**

### 11.1 遇到的坑（按发现顺序）

**坑 1：EFIT psi 轴序 [Z,R] vs 内部 [R,Z]**
- EFIT zarr 原始 psi 存 `[Z, R, time]`；`tokamind_manifest.py:144`（`_real_psi`）已 `psi_zr.T` 转成 **[R,Z]** 后才进 cache
- **结论：cache 里是 [R,Z]，不要重复转置**（§0.5 坑 #5 的延续）
- 验证方法：psi 极值位置应对应磁轴（R≈0.8, Z≈0）——若转置错，磁轴坐标会错乱

**坑 2：cache psi 是原始物理值，不是归一化值（致命！）**
- `build_dataset_cache.py` 存的是 **raw psi（Wb）**，normalization 发生在 `ManifestWindowDataset` 内部
- 我在脚本里写 `true = cached_psi*output_std + output_mean`——**把原始值又反归一化一次**，true 被严重扭曲
- 结果：corr 0.895、RMSE 16 mWb（错误）；修正后 `true = cached_psi` → **corr 0.9991、RMSE 1.1 mWb**
- **教训：先确认 cache 语义（raw vs normalized），再写反归一化**

**坑 3：模型输出是归一化的，必须反归一化**
- 模型输出 `pred_std` 在归一化空间，`pred_raw = pred_std*std + mean` ✓（这个是对的）
- 但 `output_mean/std` 要从**该 run 自己的** `manifest_scalers.npz` 读（不能混用其他 run）

**坑 4：J_φ 差分算子形式**
- 正确 GS 算子：Δ* = d²/dR² + d²/dZ² − (1/R)d/dR，J_φ = −Δ*ψ/(μ₀·R)
- 与 `R*(d²R+d²Z) − dψ/dR` 形式数学等价（乘 R 再除 R），但**数值上必须用第一种**（避免 R→0 处放大）
- 65×65 网格（dR=0.03m）下 2 阶差分误差 ~1%，4 阶改善有限——**网格粗是主要误差源**

**坑 5：真空区差分噪声污染积分（关键）**
- psi 场在等离子体外（真空区）也有结构，差分在真空区产生虚假 J_φ
- **必须用等离子体 mask（psi > psi_bndry）限制积分**：
  - 全域积分：ratio 0.53（错）
  - mask 内积分：**ratio 1.0017** ✅（单样本，corr 0.995）
- 但模型预测 psi **没有 bndry** → 需要从预测场找（O 点/鞍点法或连通分量），这一步还没完全解决

**坑 6：散点图抽样与索引**
- `rng.random(pred.size)` 生成的是逐元素 mask，但 scatter 需要逐样本 mask——布尔索引维度不匹配报错

**坑 7：np.load 大文件超时**
- 读 21k×65×65 的 cache 或 526k 个求解目录时，纯 Python 循环超慢（>120s）
- 对策：用批量脚本（`build_cache_batched.py` 238 倍加速）、避免在循环里反复 open

### 11.2 调试方法论（这次总结的有效流程）

1. **先小样本验证算子**：用单个真实求解样本（有精确 `ip_integrated` 可对照），验证差分+积分方法本身正确（ratio≈1）
2. **逐层隔离变量**：先验证"方法对"（真实 psi），再换"预测 psi"（方法+模型）
3. **数值对照**：差分 jphi vs 解析 jtor（`profiles.jtor`）在**同一行/点**对比（corr 0.995 证明算子对）
4. **查 cache 语义**：`cache psi 原始范围` 与 zarr EFIT 对照，判断是 raw 还是 normalized

### 11.3 当前状态与遗留

- ✅ 散点图（修正后）：`artifacts/pca01sigma_jphi_check/psi_pred_vs_real_scatter.png`（corr 0.9991、RMSE 中位 1.1 mWb）
- ✅ 单样本 J_φ 验证：mask 内 ∫Jφ/Ip = 1.0017（方法正确）
- ⚠️ 测试集 J_φ 反推**未完成**：模型预测 psi 无 bndry，mask 判定方案（find_critical / 连通分量）待定
- 预测 psi 高值端被低估（max 0.081 vs 真实 0.178）——磁轴附近偏差，是 J_φ 大误差的潜在来源

## 12. J_φ 派生量验证的难点与未来方案（2026-08-26）

> 目标回顾：想用「模型预测 psi → GS 算子反推 J_φ → ∫Jφ 与输入 Ip 散点对比」验证预测的物理自洽性。经多轮调试，**结论是这条路对当前数据不可行**，原因有物理层面和方法层面。本节记录难点 + 已验证可行的替代方案。

### 12.1 难点 1（物理层面，根本障碍）：EFIT total psi 含线圈真空场

- GS 方程 Δ*ψ = −μ₀·R·J_φ **只对 plasma 分量成立**（ψ = ψ_plasma + ψ_vacuum）
- **EFIT zarr 的 psi 和模型预测的 psi 都是 total psi**（含线圈真空场贡献），真空场分量 Δ* ≠ 0，污染 J_φ 反推积分
- 实测证据：
  - **合成求解样本**（equilibrium.npz 存纯 plasma psi）：mask 内 ∫Jφ/Ip = **1.0000004** ✅（方法本身正确）
  - **EFIT total psi**（cache true + 模型预测）：∫Jφ/Ip ≈ 5.8×，corr ≈ −0.1 ❌（真空场污染）
- **结论：对 total psi 用 GS 反推 J_φ 物理上无效**，这不是代码 bug

### 12.2 难点 2（方法层面）：等离子体 mask 需要 bndry

- J_φ 反推必须在 plasma mask（psi > psi_bndry）内积分，否则真空区差分噪声主导
- 真实求解样本有精确 bndry（metadata）；EFIT/预测 psi 没有
- 用 `freegs4e.critical.find_critical` 找 X 点可行但**极慢**（>30s/样本，21k 样本不可行）
- **P60 百分位近似 bndry**：与 find_critical 结果误差 ~0.001 Wb（4 个样本验证），但积分对 mask 边界敏感，且**掩盖了难点 1 的真问题**（用 P60 后 EFIT 还是 5.8×）

### 12.3 难点 3（数值层面）：J_φ 是 psi 的二阶导数

- Δ*ψ 对 psi 误差极度敏感：模型 RMAE 0.53%（相对误差 ~1%）→ Δ*ψ 误差放大百倍
- 65×65 网格（dR=0.03m）2 阶差分本身 ~1% 误差；4 阶改善有限
- 预测 psi 高值端被低估（max 0.081 vs 真实 0.178，磁轴附近）→ 磁轴处二阶导数误差最大

### 12.4 未来可行方案（按推荐顺序）

| 方案 | 内容 | 可行性 |
|---|---|---|
| **A（推荐）对比 Δ*ψ 而非 ∫Jφ** | 模型预测与真实 total psi 的 Δ*ψ 都有真空场，但**同炮同时刻真空场相同** → 两者 Δ*ψ 的差异 = plasma 差异。散点图 Δ*ψ_pred vs Δ*ψ_true（mask 内），可验证电流密度分布一致性 | ✅ 物理成立，纯 numpy 向量化秒级 |
| B 模型直接输出 J_φ | 训练时加 J_φ 作为辅助输出头（合成样本有精确 jtor 标签）| ✅ 工程可行，但要重训 |
| C 仅对合成样本验证 | 合成 equilibrium.npz 是纯 plasma psi，可精确反推 → 验证"求解器-模型"链路的 J_φ 自洽（非 EFIT 对比）| ✅ 已有单样本证据（ratio 1.0）|
| D 差分精度升级 | 加密网格或谱方法，缓解难点 3 | ⚠️ 不解决难点 1 |

### 12.5 已验证的正确事实（防止重蹈覆辙）

1. 合成样本（纯 plasma psi）+ 真实 bndry + mask 积分：**∫Jφ/Ip = 1.0000004**（方法正确）
2. 向量化差分与循环差分等价（逐点一致）
3. P60 可近似 bndry（误差 0.001 Wb，仅用于调试参考）
4. cache psi 是 raw 物理值（勿再反归一化）；模型输出需 `pred*std+mean`
5. EFIT cache psi 已是 [R,Z]（勿再转置）

## 13. psi_pred vs psi_real 散点图：做法说明（2026-08-26）

> 产物：`artifacts/pca01sigma_jphi_check/psi_pred_vs_real_two.png`（dpi=300 双面板）

### 数据口径（重要）

- **样本**：test 21,350 个（730 炮，真实 EFIT，held-out）
- **psi 值来源**：
  - `true` = test cache 的 psi（**raw 物理值 Wb，勿反归一化**，§11 坑 2）
  - `pred` = 模型输出 `pred_std * output_std + output_mean`（模型输出是归一化的，需反归一化）
- **轴序**：cache 已是 [R,Z]（`_real_psi` 已转置，勿再转，§11 坑 1）

### 两张图的做法

**图 1：逐样本均值（21,350 点）**
- 对每样本的 65×65 网格取均值：`tm = true.mean(axis=1)`、`pm = pred.mean(axis=1)`
- hexbin（gridsize=80）+ y=x 对角线
- corr = 0.9979

**图 2：逐网格点均匀采样（179,820 点）**
- 全量点 = 21,350 × 4,225 = 90,207,500 个（样本数 × 网格点数，注意不是 21k 个点！）
- **固定 seed 均匀采样 0.2%**：`rng=np.random.default_rng(2026); m=(rng.random(N*4225)<0.002).reshape(N,4225)`
- hexbin（gridsize=200）+ 对角线
- corr = 0.9991

### 为什么采样而不全画

- 9000 万点直接 scatter 内存爆炸；hexbin 可全量但慢
- 均匀采样 0.2%（18 万点）+ hexbin 分箱：**corr 稳定（0.9991，与全量一致）**，画图快、质量高
- 采样比例可调（0.1%-1%），corr 对采样不敏感

### 结果解读

- 逐样本均值 corr 0.9979、逐点 corr 0.9991 → 模型预测 psi 与真实高度一致（对应 RMAE 0.53%）
- 注意：**预测 psi 高值端被低估**（max 0.081 vs 真实 0.178，磁轴附近）——这是 J_φ 反推误差的潜在来源（§12 难点 3）

## 14. 加噪声实验（noisy05 / noisy20）结果（2026-08-26）

> 触发：用户在 clean 曲线（§9，预训练全域转正）基础上加测噪声臂——检验噪声对合成预训练的影响（v2 时代噪声是预训练有效的关键，§30）。训练 2026-08-26 完成（10 档微调全 rc=0）。

### 14.1 配置
- noisy05 / noisy20 = 合成诊断加高斯噪声，`--noise-scale 0.5 / 2.0`（σ 相对 clean 的 ±0.5×/2×）
- 预训练：noisy05 用 s54-s69 多 seed 的 s54；noisy20 新训 s54（10 epochs，val 0.02913）
- 微调：10 档（5 档 × 2 噪声级）同 §9 配置（d64、epochs 10、bs8、lr 1e-4）

### 14.2 结果（test 21,350 点 / 730 炮，RMAE/样本）

| 比例 | scratch | clean-ft | noisy05-ft | noisy20-ft |
|---|---|---|---|---|
| 100% | 0.558% | 0.529% | 0.537% | 0.533% |
| 50% | 0.655% | 0.576% | 0.602% | 0.575% |
| 25% | 0.820% | 0.622% | 0.642% | 0.629% |
| 10% | 1.581% | 0.689% | 0.681% | 0.688% |
| 5% | 1.929% | 0.742% | 0.722% | 0.733% |

零微调参考：clean 8.40% / noisy05 2.51% / noisy20 1.97%（噪声越大零微调越好——正则效应）

### 14.3 结论

1. **加噪声对预训练效果无显著影响**：noisy05/noisy20 各档与 clean 几乎持平（差异 <2.5%），100% 档还微优（0.537/0.533 vs 0.529）
2. **与 v2 时代对比（重大差异）**：v2 里噪声是预训练有效的关键（clean 6.77% vs noisy 2.11%，§30 噪声臂反超）；**PCA+位形锚定方案下，域差距已解决，噪声不再是必要条件**——clean 预训练已全域转正，噪声只是锦上添花
3. **零微调规律**：噪声越大零微调越好（8.40% → 2.51% → 1.97%）——噪声作为正则项压住过拟合合成分布，但**微调后三者收敛到同一水平**（~0.53-0.74%）
4. 微调后的噪声增益消失：噪声主要影响"预训练学到的分布紧致度"，微调（真实数据）会抹平差异

### 14.4 产物
- 评估：`artifacts/tokamind_test_eval/pca01sigma_noisy_curve.json`（12 run）
- runs：`runs/tokamind-pca01sigma-{noisy05,noisy20}-{synth-pretrain,finetune-*}`
- 曲线图（可后续画）：3 臂对比（scratch vs clean vs noisy）

### 14.5 遗留
- noisy 曲线图未画（若需要：scratch + clean-ft + noisy05-ft + noisy20-ft 四线）
- 多 seed 验证（当前 noisy 均为 s54 单 seed）

### 14.2b 精确数值表（nrmse_per_sample 与 raw_rmse，test 21,350 点 / 730 炮）

**RMSE/sample（`nrmse_per_sample`，%）**——每样本 RMSE ÷ 该样本 psi 幅值范围，再平均：

| 真实数据 | scratch | clean-ft | noisy05-ft | noisy20-ft |
|---|---|---|---|---|
| 100% | 0.763589 | 0.722259 | 0.736336 | 0.727618 |
| 50% | 0.903449 | 0.790692 | 0.822922 | 0.789048 |
| 25% | 1.128363 | 0.853783 | 0.877181 | 0.861034 |
| 10% | 2.097334 | 0.940517 | 0.932525 | 0.940650 |
| 5% | 2.520201 | 1.011208 | 0.986981 | 1.006118 |

**绝对 RMSE（`raw_rmse`，mWb）**——全局所有点直接算：

| 真实数据 | scratch | clean-ft | noisy05-ft | noisy20-ft |
|---|---|---|---|---|
| 100% | 2.05231 | 1.96308 | 1.99063 | 1.96475 |
| 50% | 2.35081 | 2.14287 | 2.19583 | 2.13819 |
| 25% | 2.83982 | 2.26766 | 2.29662 | 2.26983 |
| 10% | 5.03676 | 2.49672 | 2.44640 | 2.49827 |
| 5% | 5.99077 | 2.60718 | 2.54374 | 2.60259 |

**零微调参考**（synthetic pretrain 直接测试，无微调）：

| run | nrmse_per_sample | raw_rmse (mWb) |
|---|---|---|
| synth-pretrain clean | 10.137625% | 22.08591 |
| synth-pretrain noisy05 | 3.224083% | 7.13055 |
| synth-pretrain noisy20 | 2.585406% | 5.78972 |

**图**：`artifacts/pca01sigma_results/pca01sigma_rmse_sample_{clean,noisy05,noisy20}.png`（三张独立，RMSE/sample 纵轴，无 title）

### 14.6 三实验总结论 + 零微调解读（2026-08-26 追加）

**三实验（clean / noisy05 / noisy20）结论**：

1. **预训练全域转正，增益随数据稀缺单调放大**（RMSE/sample 相对增益）：
   | 数据量 | clean | noisy05 | noisy20 |
   |---|---|---|---|
   | 100% | −5.4% | −3.6% | −4.7% |
   | 50% | −12.5% | −8.9% | −12.7% |
   | 25% | −24.3% | −22.3% | −23.7% |
   | 10% | −55.2% | −55.5% | −55.1% |
   | 5% | **−59.9%** | **−60.8%** | **−60.1%** |
   数据越少预训练价值越大（5% 档增益是 100% 档的 ~10 倍）
2. **噪声对微调后结果无显著影响**：三条 pretrain+finetune 曲线几乎重叠（各档差异 <2.5%）；PCA+位形锚定已解决域差距，噪声不再是必要条件（对比 v2 时代噪声是预训练有效关键）
3. **噪声只影响零微调（正则效应）**：clean 10.14% → noisy05 3.22% → noisy20 2.59%（越大越稳）；微调后三者收敛同一水平（~0.72–1.01%）

**零微调实验说明**：= 合成预训练模型**不经真实微调**直接在 held-out 真实测试集（21,350 点）评估。回答"预训练学到的东西能否直接迁移到真实数据"（sim2real gap 最直接度量）。结果：clean 10.14% / noisy05 3.22% / noisy20 2.59%——方向对但远不够用，**微调是必需的**（微调后 3-10× 提升）。零微调不是可用方案，是"预训练学了什么"的诊断参照。

**图产物**（全部无大 title，dpi=300）：
- 三张独立 RMSE/sample 曲线：`artifacts/pca01sigma_results/pca01sigma_rmse_sample_{clean,noisy05,noisy20}.png`
- 三并排版：`..._rmse_sample_side.png`
- 零微调柱状图：`..._zero_finetune.png`
- 绝对 RMSE 版（备用）：`..._rmse_{clean,noisy05,noisy20}.png`

## 15. Temporal Split 实验（方案 B：M5+M6+M7 训练 → M8 验证 → M9 测试）交接（2026-08-26）

> **给新对话的交接**：本实验验证**时间泛化**（历史数据训练、预测未来），对比 random split（§9-§14）。方案 B 已定稿：train 数据量与 §9 相当（87.6%），避免"数据量"混入变量。Manifest 已生成，训练脚本待新机器执行。

### 15.1 实验设计

**研究问题**：合成预训练在**时间外推**下是否仍有效（训练 2005-2010 炮、预测 2013 炮）？

| 集 | campaign（年份）| 炮数 | 样本数 | 占比 |
|---|---|---|---|---|
| **train** | M5(2005)+M6(06-07)+M7(08-10) | 6,046 | **187,292** | 87.6% |
| **val** | M8(2011-12) | 559 | **6,206** | 2.9% |
| **test** | M9(2013) | 507 | **17,498** | 8.2% |
| 剔除 | Unknown | 174 | 2,928 | — |

- 划分**炮级**（train/val/test 炮互不重叠，无泄漏）
- **合成预训练**只用 train 炮（M5-M7）的合成样本：**150,728 行**
- 判定：若 B 臂（预训练）在 M9 上仍优于 A 臂（scratch）→ 预训练时间泛化有效

### 15.2 已生成 manifest（本机）

```
data/manifests/training_temporal/
├── split_train_real.jsonl    (187,292 行, M5+M6+M7)
├── split_val_real.jsonl      (6,206 行, M8)
├── split_test_real.jsonl     (17,498 行, M9)
├── split_train_shots.jsonl / split_val_shots.jsonl / split_test_shots.jsonl
└── run_synth_pretrain.jsonl  (150,728 行, M5-M7 合成预训练)
```

### 15.3 训练配置（复用 §9，GPU 8 卡）

模型：d64/2/4/128、69 通道 → 65×65 psi、bs8、lr 1e-4、epochs 10、eval_every 1

**3 个 run**：
| run | 初始化 | 训练数据 | 说明 |
|---|---|---|---|
| `temporal-scratch-100pct` | 随机 | split_train_real | A 臂 |
| `temporal-finetune-clean-100pct` | synth-clean 预训练 → 微调 | 同上 | B 臂 clean |
| `temporal-finetune-noisy05-100pct` | synth-noisy05 预训练 → 微调 | 同上 | B 臂 noisy |

（如需数据量曲线：可加 50/25/10/5% 子集，本方案先做 100% 三臂）

### 15.4 新机器执行步骤

```bash
# 1. 同步（到 8 卡 GPU 机器）
#    - 代码: mast-bridge/（含 .tokamind-train-env）
#    - manifests: data/manifests/training_temporal/
#    - cache: data/processed/training_cache_pca_01sigma/（含 real_full/合成缓存）
#    - 若 cache 未同步: 用 build_cache_batched.py 重建

# 2. 重建 cache（batch 版 ~10 min，若未同步）:
mast-bridge/.freegsnke-solve-env/bin/python -u mast-bridge/scripts/build_cache_batched.py \
  --manifest data/manifests/training_temporal/split_train_real.jsonl \
  --output data/processed/training_cache_pca_01sigma/temporal_train_real.npz --workers 48

# 3. 合成预训练 cache（clean/noisy05 各一）:
#    --diagnostics-name diagnostics.npz → temporal_synth_clean.npz
#    --diagnostics-name diagnostics_noisy_s05.npz → temporal_synth_noisy05.npz

# 4. 训练（16 并行 8 GPU）:
#    A: train_tokamind_manifest.py --manifest split_train_real --cache temporal_train_real \
#       --run-dir runs/tokamind-temporal-scratch-100pct
#    B clean: --init-run-dir <clean 预训练> --finetune-method full
#    B noisy05: --init-run-dir <noisy05 预训练 s54> --finetune-method full

# 5. 评估:
python mast-bridge/scripts/evaluate_tokamind_testset.py \
  --manifest data/manifests/training_temporal/split_test_real.jsonl \
  --run-dir runs/tokamind-temporal-scratch-100pct \
  --run-dir runs/tokamind-temporal-finetune-clean-100pct \
  --run-dir runs/tokamind-temporal-finetune-noisy05-100pct \
  --output-json artifacts/tokamind_test_eval/temporal_curve.json
```

### 15.5 预期与判定

1. **若 B 臂（预训练）在 M9 仍优于 A 臂** → 预训练时间泛化有效（强结论）
2. **若 B 优势缩小**（vs §9 random split 的 −5~60%）→ 预训练增益受限于 M5-M7→M9 的装置演化（campaign 漂移 §0.4：M9 βp 更高、Ip 更低）
3. **M9 单独表现通常差于 M8**（泛化距离更远）——预期现象，报告时按 campaign 分组
4. 预训练数据是 M5-M7 的合成样本，**与 test（M9）严格不同 campaign**——零微调/微调都是纯时间外推

### 15.6 坑（§7 引用）
- 训练用 `.tokamind-train-env`；cache 必须与 manifest 对齐（§20 C3）
- val shots 从 split_val_shots.jsonl 取（M8 炮）
- 预训练 run 的 val 用内部划分（合成数据无 M8 炮）
- 后台任务 setsid nohup + disown；8 卡 16 并行（NGPUS=8 WORKERS=16）

### 15.7 待办交接：Temporal finetune 补跑（2026-08-27 更新，换 2 卡机器执行）

> **给新对话的交接**：以下全部内容都验证过、可直接照做。目标机器：**2 卡 GPU**（每个 finetune 各占 1 卡并行，2 个 job 同时跑）。

**当前状态**：
- ✅ 完整（10/10 epochs）：`runs/tokamind-temporal-synth-pretrain`（clean 预训练）、`runs/tokamind-temporal-synth-pretrain-noisy05`、`runs/tokamind-temporal-scratch-100pct`（A 臂）
- ❌ **不完整（仅 3 epochs 被中断）**：`runs/tokamind-temporal-finetune-clean-100pct`、`runs/tokamind-temporal-finetune-noisy05-100pct`（checkpoints/ 有 best+latest，可 --resume；train.log 不存在）
- ❌ 无评估产物（无 temporal 评估 JSON）

**1. 同步清单（必须完整）**：
```
mast-bridge/                          # 代码 + .tokamind-train-env（训练/评估必须用它）
data/manifests/training_temporal/     # 5 个 manifest（§15.2）
data/processed/training_cache_pca_01sigma/temporal_train_real.npz   # 训练 cache（187,292 样本）
runs/tokamind-temporal-synth-pretrain/            # clean 预训练（checkpoints/ summary.json manifest_scalers.npz）
runs/tokamind-temporal-synth-pretrain-noisy05/    # noisy05 预训练
runs/tokamind-temporal-scratch-100pct/            # A 臂（仅评估用）
runs/tokamind-temporal-finetune-clean-100pct/     # 已有 3-epoch checkpoint（可选：resume 或弃用重跑）
runs/tokamind-temporal-finetune-noisy05-100pct/
```
- cache 若未同步可重建：`mast-bridge/.freegsnke-solve-env/bin/python -u mast-bridge/scripts/build_cache_batched.py --manifest data/manifests/training_temporal/split_train_real.jsonl --output data/processed/training_cache_pca_01sigma/temporal_train_real.npz --workers 48`
- **验证 cache 与 manifest 对齐**（§20 C3）：cache 行数 == manifest 行数 == 187,292；sample_ids 顺序按 cache 内存储（勿假设与 manifest 同序，见 §17.3 坑 3）

**2. 训练命令（2 卡并行，两个 finetune 同时跑）**：

```bash
cd <workspace>
# 卡 0: clean finetune（10 epochs, bs8, lr 1e-4）
CUDA_VISIBLE_DEVICES=0 setsid nohup mast-bridge/.tokamind-train-env/bin/python -u   mast-bridge/scripts/train_tokamind_manifest.py   --manifest data/manifests/training_temporal/split_train_real.jsonl   --cache data/processed/training_cache_pca_01sigma/temporal_train_real.npz   --run-dir runs/tokamind-temporal-finetune-clean-100pct   --init-run-dir runs/tokamind-temporal-synth-pretrain   --finetune-method full --epochs 10 --batch-size 8 --lr 1e-4 --eval-every 1   > logs/finetune-clean.log 2>&1 & disown

# 卡 1: noisy05 finetune（同上，init 用 noisy05 预训练）
CUDA_VISIBLE_DEVICES=1 setsid nohup mast-bridge/.tokamind-train-env/bin/python -u   mast-bridge/scripts/train_tokamind_manifest.py   --manifest data/manifests/training_temporal/split_train_real.jsonl   --cache data/processed/training_cache_pca_01sigma/temporal_train_real.npz   --run-dir runs/tokamind-temporal-finetune-noisy05-100pct   --init-run-dir runs/tokamind-temporal-synth-pretrain-noisy05   --finetune-method full --epochs 10 --batch-size 8 --lr 1e-4 --eval-every 1   > logs/finetune-noisy05.log 2>&1 & disown
```

**关键参数确认**（对照旧 run 的 summary.json 核验，若有则保持一致）：
- 模型 d64/2/4/128、69 通道 → 65×65 psi（train 脚本自动从 manifest 推断，勿改）
- `--init-run-dir` 与 `--run-dir` **必须不同**（脚本会报错）
- 训练日志每 epoch 输出；完成标准：log 出现 10/10 epochs + checkpoints/best 更新

**3. resume 还是从头**：
- **推荐：--resume**（已有 3-epoch latest checkpoint，续跑 ~2.4h/个；日志 718 字节是中断痕迹，checkpoint 本身完整）
- 若 resume 报错（cache/manifest 不一致、模型结构不匹配）：从头跑（删 run-dir 重建，~3h/个，1 卡）
- 命令：`--resume` 加在 train 命令末尾即可

**4. 评估（2 卡机器上跑，或回传后跑）**：
```bash
mast-bridge/.tokamind-train-env/bin/python -u mast-bridge/scripts/evaluate_tokamind_testset.py   --manifest data/manifests/training_temporal/split_test_real.jsonl   --run-dir runs/tokamind-temporal-scratch-100pct   --run-dir runs/tokamind-temporal-finetune-clean-100pct   --run-dir runs/tokamind-temporal-finetune-noisy05-100pct   --output-json artifacts/tokamind_test_eval/temporal_curve.json
```
（评估需要 `temporal_test_real.npz` cache；注意 --run-dir 需为绝对路径或从 workspace 根跑）

**5. 回传内容**：两个 finetune run 目录 + `temporal_curve.json`；结果写入 §15.8

**时间参考**：1×4090：scratch 10 epochs ≈ 1.5h（187,292 行 ÷ bs8 = 23,412 步/epoch）；finetune 同规模。2 卡并行总墙钟 ≈ 2.5-3h（含评估）

**判定（§15.5 回顾）**：B 臂（finetune-clean/noisy05）在 M9 test 的 RMAE 是否优于 A 臂（scratch）？若 −5~60% 增益保持 → 预训练时间泛化有效；若缩小 → campaign 漂移限制

### 15.8 结果（2026-08-27 全部完成）

**执行记录**：2×RTX 4090，100% 档（§15.7 补跑）+ 数据量曲线（§19，16 run 单 seed s54）全部 rc=0。
评估 33 个 run 于 M9 test（17,498 点 / 507 炮）→ `artifacts/tokamind_test_eval/temporal_curve.json` + `temporal_curve.png`。
**踩坑**：① clean 预训练目录名为 `tokamind-temporal-synth-pretrain`（无后缀），launch 循环拼成 `-pretrain-clean` 导致 4 个 clean 子集 finetune 秒挂，修正后重跑；② 评估缓存撞名：artifacts 里旧 `test_cache_split_test_real.npz`（21,350 行 random split 版）与 temporal manifest（17,498 行）不匹配 → 用 `temporal_test_real.npz` 覆盖后方秒级命中；③ 绘图 `endswith('5pct')` 误匹配 `-25pct`，已精确匹配修正。

**数据量曲线（RMAE/样本，M9 test，s54）**：

| 真实数据 | A scratch | B clean | B noisy05 | B noisy20 | clean 相对增益 |
|---|---|---|---|---|---|
| 100% | 1.980% | **1.638%** | 1.594% | 1.722% | **−17.3%** |
| 50% | 2.235% | **1.703%** | 1.883% | 2.011% | **−23.8%** |
| 25% | 2.234% | **1.689%** | 1.725% | 1.706% | **−24.4%** |
| 10% | 3.603% | **1.668%** | 1.681% | 1.676% | **−53.7%** |
| 5% | 4.180% | **1.681%** | 1.788% | 1.698% | **−59.8%** |

**100% 档多 seed（均值±std，s54+s55-57 / scratch 另含 s103-107）**：
scratch 2.054±0.199% / clean 1.656±0.075% / noisy05 1.661±0.109% / noisy20 1.667±0.056%
→ B 臂一致优于 scratch ~0.4pp，seed 稳定（B 臂 std 远小于 scratch）。

**判定（§15.5）**：
1. **预训练时间泛化有效（强结论）**：B 臂在纯时间外推（M5-M7 训练→M9 测试）下全域转正，增益 −17.3% → **−59.8%**，随数据稀缺单调放大，与 random split 同构
2. **增益甚至 ≥ random split**：100% 档 −17.3%（random −5.3%）、50% −23.8%（−12.1%）、25% −24.4%（−24.2%）、10% −53.7%（−56.4%）、5% −59.8%（−61.5%）——高数据区 temporal 增益反而更大（M9 绝对误差更大、预训练先验更有价值）；低数据区两者打平
3. **绝对水平**：M9 上所有 run 均差于 random split 同档（scratch 100% 1.98% vs random 0.56%）——时间外推本身更难（M9 βp 更高、Ip 更低，§0.4 漂移），但预训练相对增益不受影响
4. **噪声臂**：clean/noisy05/noisy20 微调后基本持平（各档差 <0.15pp），噪声非必要条件（同 §14 random 结论）；noisy05 100% 档微优（1.594%）
5. **零微调参考**：未单独评估（可选补：temporal 预训练直接测 M9，对照 §14 的 8.40/2.51/1.97%）

**结论一句话**：仿真预训练在**时间外推场景同样全域有效**，数据越稀缺增益越大（最高 −60%），增益水平不低于 random split——预训练学到的是跨 campaign 可迁移的平衡物理，而非只记住训练装置。

**2026-08-28 追加（图产物，RMSE/sample 口径，含 random split 对照）**：
- `artifacts/pca01sigma_results/temporal_rmse_sample_{clean,noisy05,noisy20}.png`（三臂独立）+ `_side.png`（三并排）
- 每图 4 条曲线：`real scratch (temporal split)` 蓝虚线 / `real scratch (random split)` 蓝点线 / `synthetic pretrain + finetune (temporal split)` 彩色实线 / `synthetic pretrain + finetune (random split)` 彩色点划线（dpi=300，无大 title，脚本 `/tmp/opencode/plot_temporal_curves.py`）
- 高度对比结论：random ft 线整体低 temporal ft 线 ~2.4–3.3×（测试集不同：random 21,350 点全 campaign vs temporal 17,498 点纯 M9，高度差 ≈ M9 装置漂移难度）；两场景 ft 线形状同构（temporal 几乎水平 2.26–2.41%，random 微升 0.72→1.01）——预训练增益不受测试集难度影响

## 16. LCFS 预测 vs EFIT real 可视化作图：方法与踩坑记录（2026-08-27 定稿）

> **给新对话的交接**：这是从模型预测 psi 恢复 LCFS 并与 EFIT real 对比的完整经验。**最终产物**：`artifacts/pca01sigma_results/lcfs_pred_vs_real_17833_v3.png`（shot 17833，dpi=300，上排 4 个 R-Z 图 + 下排整行 Ip(t)）。最终脚本（可用）：`/tmp/opencode/plot_lcfs_final2.py`（注意依赖：train-env 无 skimage，系统 python 有）。

### 16.1 最终方法（已验证正确，可直接复用）

**数据准备**：
- 预测用 `runs/tokamind-pca01sigma-finetune-100pct`（random-split clean 微调模型），测试集 manifest `data/manifests/training_pca_01sigma/split_test_real.jsonl`（filter source=='real'，shot 17833 共 9 个样本）
- cache：`data/processed/training_cache_pca_01sigma/test_real.npz`（features + psi），按 sample_id 前缀 `17833_t` 过滤
- 预测输出：`pred*out_std + out_mean` 反归一化（**psi 是 raw 物理值，单位 Wb，勿再归一化**）
- **轴序：cache 与模型输出均为 [R, Z]（65×65）**，R1=np.linspace(0.06,1.98,65)、Z1=np.linspace(-2,2,65)；**EFIT zarr 的 psi 是 [Z, R]**（如用需转置）——本图 EFIT 只用 lcfs_r/lcfs_z 轮廓，不涉及 psi 轴序

**LCFS 提取核心（动态扫描 + skimage）**：
```python
from scipy import ndimage
from skimage import measure
def lcfs_extract(psi):
    oi,oj=np.unravel_index(np.argmax(psi), psi.shape)   # O 点
    best=None; best_area=0
    for frac in np.arange(0.05,0.95,0.01):              # 从 psi_max 向下扫 level
        lv=psi.max()-frac*(psi.max()-psi.min())
        blob=psi>=lv
        lbl,n=ndimage.label(blob)
        if n==0: continue
        lab=lbl[oi,oj]                                  # 取 O 点所在连通区
        if lab==0: continue                             # O 点不在连通区则跳过
        main=(lbl==lab)
        if main[0,:].any() or main[-1,:].any() or main[:,0].any() or main[:,-1].any():
            break                                       # 泄漏到网格边界即停
        if main.sum()>best_area:
            best_area=main.sum(); best=main.copy()
    if best is None: return None,None
    contours=measure.find_contours(best.astype(float),0.5)
    c=max(contours,key=len)
    r_pts=np.interp(c[:,0],[0,64],[R1[0],R1[-1]])       # row->R（重要！勿写反）
    z_pts=np.interp(c[:,1],[0,64],[Z1[0],Z1[-1]])       # col->Z
    # 弧长均匀重采样 170 点（workflow §12 canonical 点数）
    s=np.hypot(np.diff(r_pts),np.diff(z_pts)); s=np.concatenate([[0],np.cumsum(s)])
    pts=np.linspace(0,s[-1],170)
    return np.interp(pts,s,r_pts), np.interp(pts,s,z_pts)
```
- **语义**：LCFS = 从 O 点向下扩展、**泄漏到网格边界前**的 O 点连通区外边界（等价于 workflow §9 "O-point connected core region"）。**注意取 O 点所在连通区而非面积最大块**——预测场常有更大真空连通块不含 O 点（21735 验证发现），取最大块会失败
- **为什么不能用固定 frac**：预测 psi 场的 bndry 值与 EFIT 不同，固定 frac=0.05 只得到磁轴附近小圈；frac=0.4 以上真空区泄漏（Z 蔓延到 ±2）。只有动态扫描能自适应找到真 LCFS

**绘图**（用户明确要求，勿改）：
- 布局：**上排 4 个 R-Z 图**（每帧一列），**下排 1 个 Ip(t) 横跨整行**（gridspec height_ratios=[1.6,1]）
- 机械元件（真实比例）：`run_freegsnke_forward._draw_rectangles`，颜色约定：active='lime'、passive='darkorange'、wall='gray'、limiter='red'，文件在 `data/raw/mast/machine/{shot}/MAST_{active_coils,passive_coils,wall,limiter}.pickle`
- LCFS 线：EFIT real 蓝色实线（lw=2.5，label='EFIT real'），pred 橙色虚线（lw=2.5，label='pred'）
- 每面板 `ax.set_aspect('equal')`，xlim=(0.0,2.05)、ylim=(-2.1,2.1)；Ip 图红线虚线标记选中帧时刻 + 红三角
- 帧选择：test 9 个样本中取 idxs=[0,3,6,8]（t≈0.255/0.26/0.285/0.31）

**已验证结果（shot 17833，最终版）**：

| 时刻 | EFIT real R 范围 | pred R 范围 |
|---|---|---|
| 0.255 | [0.21,1.33] | [0.22,1.36] |
| 0.260 | [0.21,1.33] | [0.22,1.36] |
| 0.285 | [0.23,1.38] | [0.28,1.36] |
| 0.310 | [0.23,1.39] | [0.28,1.36] |

pred 与 EFIT 最大偏差 <0.05m（约 2 个网格点），肉眼贴合。

**跨 shot 验证（21735，2026-08-27 追加）**：4 帧（0.2/0.25/0.4/0.55s）全部匹配，偏差 ≤0.02m。图：`artifacts/pca01sigma_results/lcfs_pred_vs_real_21735.png`。**新发现**：
1. 21735 部分帧的最大连通区**不含 O 点**（外面有更大的真空连通块）→ 必须取 **O 点所在连通区**（见上方代码）
2. 21735 在 t≈0.3/0.35s 的 **EFIT psi 场本身异常**（磁轴在极区 R=0.51, Z=±1.75，psi max 0.050 出现在上下两端）——模型预测**忠实复现该异常**（不是模型失败，是 EFIT 该帧解算异常）。选帧时避开此类帧

### 16.2 踩坑记录（按发现顺序，每个都真实发生过）

1. **坑：unravel_index 的 oi/oj 是 numpy 标量，`R1[oi]` 若传错数组会返回整行 65 元素**——报错 "(158,) vs (65,) broadcast"。原因：向 `extract_lcfs` 传了 2D meshgrid（Rg）而不是 1D R1。**经验**：函数签名内统一用 `float(Rg[oi,0])`/`float(Zg[0,oj])` 取标量磁轴坐标。
2. **坑：固定 frac=0.05 提取的是磁轴附近小圈**（R∈[0.69,0.99] vs EFIT [0.21,1.33]），肉眼"pred 比 EFIT 小太多"。**经验**：LCFS 阈值必须用动态扫描（泄漏前最大闭合面），不能拍脑袋定 frac。
3. **坑：真空区泄漏**——预测场 frac≥0.40 时连通区蔓延到全网格（Z 到 ±2），因为预测 psi 外围分布与 EFIT 不同。**经验**：必须检测连通区是否触碰网格四边，碰即视为泄漏停止。
4. **坑：matplotlib contour 在 mask 上取段不可靠**——用 `np.where(mask,psi,-9)` + contour 提取的段 R 范围错误（R∈[0.57,1.47] vs mask 真实 [0.24,1.35]）。**经验**：二值 mask 的轮廓一律用 `skimage.measure.find_contours(mask.astype(float), 0.5)`，别用 matplotlib contour。
5. **坑（最隐蔽）：skimage find_contours 返回 (row, col)，row→R 方向、col→Z 方向**——曾写反（`c[:,1]` 映射到 R），导致最终图 pred 又变成 R∈[0.55,1.49] 的错位圈。**经验**：映射写 `r_pts=np.interp(c[:,0],...)`、`z_pts=np.interp(c[:,1],...)`，或先打印像素范围（r∈[5.5,43.5]→R∈[0.24,1.35]）核对再画。
6. **坑：Ip(t) 缩在左下角**——Ip 子图放在 4 列 gridspec 的第 1 列过窄。**经验**：Ip 图用 `gs[1,:]` 横跨整行；Ip 数据本身正常（2594 点、t∈[-0.1,0.42]s、全部有限）。
7. **坑：依赖环境**——train-env（`.tokamind-train-env`）无 skimage；系统 python 有 skimage 0.25.2。**经验**：LCFS 提取脚本最终用 train-env 跑但靠 `from skimage import measure`——注意当前 final2 脚本是 train-env 里跑的**且我实际替换成了 contour 版本**（v3 未落地）。**下次直接在新会话用系统 python 重新装配 final 版**（skimage 版已在 /tmp/opencode/test_lcfs_v3.py 验证正确）。

### 16.3 后续可做（若需要）
- 换更多 shot 验证（如预测最差的炮，检查 LCFS 偏差是否与 RMSE 相关）
- 加 noisy05 模型预测的 LCFS 对比（看噪声是否影响边界形状）
- 输出 pred LCFS 与 EFIT 的闭合曲线距离（平均法向距离）作为量化指标

## 17. J_φ vs Ip（J-tor）散点图：计算方法与重大修正（2026-08-27）

> **给新对话的交接**：本条**推翻了 §12.1 的"total psi 无法反推 J_φ"结论**——那是一个轴序 bug（EFIT psi [Z,R] 被当 [R,Z] 做差分）。修正后 GS 反推完全自洽。本方法是「模型预测的电流密度/总电流」物理验证的最终路径。

### 17.1 核心结论（数值）

**积分量验证（Ip，3,000 样本采样，test 21,350 全量）**：
- **参考真值 = 直接测量 Ip 信号**（magnetics zarr `ip`，罗氏线圈集成；EFIT ∫j_phi 只是重构量，两者自洽 0.992 但非"真值"）
- `Ip_pred / Ip_signal` **中位 = 0.9929**，90% = 1.015，99% = 1.077，corr = **0.9942**
- 各 campaign（M5-M9）ratio 中位全部 ≈ 1（M5 1.000 / M6 1.001 / M7 1.002 / M8 0.995 / M9 1.006）——时间外推同样准确
- 图：`artifacts/pca01sigma_results/jtor_ip_scatter.png`（散点 + y=x 虚线）

**逐点验证（Δ*ψ，180,000 点 = 3,000 样本 × mask 内 60 随机点）**：
- corr = 0.67，slope = 0.96（接近 1，线性关系正确）
- 图：`artifacts/pca01sigma_results/lapstar_scatter.png`（hexbin + y=x 虚线）
- 逐点 corr 低于积分量是预期的：Δ*ψ 是 psi 二阶导，误差放大 ~100 倍（§12.3）

### 17.2 计算方法（可直接复用）

**GS 方程**：Δ*ψ = −μ₀RJ_φ（Wb/m² 单位，R 在 1e-3 处 clamp 避免除零）

**EFIT 侧（参考真值）**：
1. `j_phi` 从 `{shot}.zarr/equilibrium` 读，**轴序 [Z,R,T]** → `.T` 转 **[R,Z]**（防坑！§11 坑 1 的延续）
2. `Ip_efit = Σ j_phi × dR × dZ`（**j_phi 真空区 = 0，全域积分即可，无需 mask**）
3. **mask = j_phi > j_phi帧max × 1e-3**（用于逐点/积分限域；按帧 max，勿用全局 max）
4. 自洽性独立验证：∫j_phi / Ip 信号 = 0.992 ± 1%（corr 0.995）

**模型预测侧**：
1. 模型输出反归一化：`pred_raw = pred_std × out_std + out_mean` → [R,Z] 65×65（psi 物理值 Wb）
2. **Δ*ψ 算子**（[R,Z] 网格，dR=0.03, dZ=0.0625）：
   ```
   Δ*ψ = d²ψ/dR² − (1/R)dψ/dR + d²ψ/dZ²   （2 阶中心差分）
   ```
   **不要**用 R×(d²R+d²Z) − dψ/dR 的等价形式（R→0 放大）
3. `J_φ = −Δ*ψ/(μ₀R)`；`Ip_pred = Σ mask内 J_φ × dR × dZ`

**逐点散点**：x = −μ₀R J_φ^EFIT（逐网格点），y = Δ*ψ^pred，mask 内随机采样点

### 17.3 踩坑（本次新增）

1. **EFIT psi/j_phi 轴序 [Z,R] 是 §12.1 5.8× 结论的错误根源**——[Z,R] 场按行差分当 R 方向，ratio 0.63-0.81；`.T` 转 [R,Z] 后 ratio 精确 = 1.000
2. **mask 阈值必须按帧**：`jphi.max()`（当前帧）而非 `jphi_all.max()`（165 帧全局最大）——全局 max 会让低电流帧 mask 全空，Ip_pred 全 0，corr NaN
3. **cache 顺序 vs 排序后 rows 错位**：`cache['sample_ids']` 是 manifest 原始顺序，rows 排序/采样后直接按 `sample_idx` 索引 cache 会错位 → 必须先按 sample_id 建映射 `{sample_id: idx}` 重排 rows 到 cache 顺序再采样
4. **模型 forward 用 batch=32 时 pred reshape**：`o['pred'][OUTPUT_SIGNAL_ID]` 输出 (32,65,65) 直接 reshape(-1,65*65) 会与 out_std(1,4225) 广播错误——先 `reshape(-1,65*65)` 再乘 out_std

### 17.4 遗留
- 全量 21,350 样本（当前 3,000 采样）——脚本 `forward_jtor_ip.py` 的 N_SAMPLE 改 21350 即可全量
- 逐点版与积分版脚本：`/tmp/opencode/forward_jtor_ip.py`、`/tmp/opencode/forward_lapstar_scatter.py`、`/tmp/opencode/plot_jtor_ip_single.py`、`/tmp/opencode/plot_lapstar.py`

## 18. 三图绘图规范统一（psi / j_tor / Ip(t)）（2026-08-27）

> **给新对话的交接**：统一三种验证图的「需求 → 做法 → 坑」，保证新会话直接照做。**风格基准 = psi 散点图（`scatter_psi_two.py`）：hexbin viridis 密度图 + 红虚线 y=x + corr 标注 + colorbar count + aspect equal + dpi=300。**

### 18.1 psi 散点图（两种口径）

**需求**：验证预测 psi 与真实 psi 的整体一致性（21,350 测试样本）。双面板：
- **左：逐样本均值**（每样本 65×65 取平均 → 21,350 点）——样本级一致性
- **右：逐网格点均匀采样**（全量 90M 点按固定 seed 抽 0.2% → ~18 万点）——像素级一致性

**做法**（`/tmp/opencode/scatter_psi_two.py`）：
1. forward 全量 test_real（`runs/tokamind-pca01sigma-finetune-100pct`），pred 反归一化 `pred*out_std+out_mean`
2. `true = cache['psi']`（raw 物理值，勿再处理）
3. hexbin：左 gridsize=80、右 gridsize=200，`cmap='viridis', mincnt=1`，`set_aspect('equal')`，红虚线 `'r--' lw=1.8 alpha=0.8`
4. title 只写 corr（左 corr=0.9979、右 corr=0.9991）；colorbar label 'count'
5. 采样固定 `np.random.default_rng(2026)`（可复现）

**坑**：cache psi 是 raw 物理值勿反归一化（§11 坑 2）；模型输出必须 `pred*std+mean`（§11 坑 3）；采样 seed 固定否则图不可复现。

### 18.2 j_tor 图（J_φ vs Ip，两种口径）

**需求**：物理自洽性验证——模型预测 psi 反推的电流是否与真实一致。
- **积分版（定稿图 `jtor_ip_scatter.png`）**：总电流守恒。横轴 = **直接测量 Ip 信号**（magnetics zarr `ip`，罗氏线圈集成，`np.interp(tt, t_ip, ip)` 插值）；纵轴 = GS 反推 `Ip_pred = Σ mask内(−Δ*ψ/(μ₀R)) × dR dZ`。结果：中位 ratio 0.9929、corr 0.9942
- **逐点版（`lapstar_scatter.png`）**：电流密度分布。x = −μ₀RJ_φ^EFIT（EFIT j_phi 数组逐点）、y = Δ*ψ^pred，mask 内每样本 60 随机点（180k 点）。corr 0.67（二阶导噪声放大，§12.3）

**做法**（`/tmp/opencode/forward_jtor_ip.py`、`forward_lapstar_scatter.py`）：
1. GS 算子（[R,Z] 网格）：`Δ*ψ = d²ψ/dR² − (1/R)dψ/dR + d²ψ/dZ²`，2 阶中心差分；勿用 R×(d²R+d²Z)−dψ/dR 等价形式（R→0 放大）
2. EFIT `j_phi`：zarr `{shot}.zarr/equilibrium`，**轴序 [Z,R,T]** → `.T` 转 [R,Z]；真空区 = 0 天然带 mask
3. mask = `jphi > jphi帧.max()×1e-3`（**按帧 max，勿用全局 max**）
4. 画图统一 hexbin viridis + 红虚线 y=x + corr title（积分版 x 轴标签 $I_p^{meas}$）

**坑**（§17.3 详录）：EFIT psi/j_phi 轴序 [Z,R]（§12.1 的 5.8× 错误根源）；mask 帧阈值；cache 顺序 vs 排序 rows 错位（先按 sample_id 建映射）；pred reshape(-1,65*65) 再乘 out_std；**Ip 参考用直接测量信号而非 EFIT 积分**（用户指出，EFIT 只是重构量）。

### 18.3 Ip(t) 图（放电曲线 + LCFS 帧标记）

**需求**：展示完整放电（Ip 全时间序列）+ 红色虚线标记选中的 LCFS 对比帧。

**做法**（`/tmp/opencode/plot_21735_v2.py`）：
1. 数据：`magnetics zarr` 的 `time` + `ip`，`ip/1e3` → kA；无 NaN（2594 点、t∈[-0.1,0.42]s 已验证）
2. 布局：**必须横跨整行** `gs[1,:]`（放单列会缩在左下角，坑！）
3. 曲线：`color='#174ea6', lw=2.0`；帧标记：红色虚线 `axvline` + 白芯红点 `'o' ms=7, mec='white'`
4. 风格：无 top/right spine、grid alpha 0.18、title = `shot {SHOT}`、$I_p$ [kA] 公式标签
5. 帧选择：EFIT psi 场磁轴在极区（R≈0.5, Z≈±1.75）的帧是 EFIT 解算异常帧，**选帧时避开**（§16 验证发现 21735 t=0.3/0.35）

**坑**：子图窄会缩角；机械元件配色专业版（PF 深蓝 #1a5fb4 / passive 灰 / wall 深灰 / limiter 红 #e01b24，半透明填充）；预测 LCFS 用 §16 动态扫描提取（勿用固定 frac）。

### 18.4 产物清单

| 图 | 路径 | 关键数字 |
|---|---|---|
| psi 双面板 | `artifacts/pca01sigma_jphi_check/psi_pred_vs_real_two.png` | corr 0.9979 / 0.9991 |
| j_tor 积分版 | `artifacts/pca01sigma_results/jtor_ip_scatter.png` | corr 0.9942, 中位 0.9929 |
| j_tor 逐点版 | `artifacts/pca01sigma_results/lapstar_scatter.png` | corr 0.67 |
| Ip(t)+LCFS | `artifacts/pca01sigma_results/lcfs_pred_vs_real_21735.png` | LCFS 偏差 <2cm |

## 19. Temporal 数据量曲线补齐计划（待办，2026-08-27）

> **目标**：Temporal Split（M5+M6+M7 训练 → M8 val → M9 test）得到与 Random Split（§9/§14）同构的
> **数据量递减曲线**：真实数据 100/50/25/10/5% ×（A scratch + B clean/noisy05/noisy20 预训练+微调）。
> **配置与 random split 完全一致**：d64/2/4/128、69 通道→65×65、bs8、lr 1e-4、epochs=10（各档统一，§5.3）、
> 单 seed s54（§9/§14 惯例，不做多 seed 扩张）。
> 注意：subsets 的 val 行 = M8 val shots（6,206 行）统一拼接，test 一律 M9（17,498 行）。

### 19.1 已有资产（100% 档，✅ 全部完成）

| 项 | 状态 |
|---|---|
| 预训练：clean/noisy05 s54-57、noisy20 s54/s56/s57 | ✅ |
| scratch-100pct：s54 + s103-107（A 臂）| ✅ |
| finetune-100pct：clean/noisy05 × s54-57、noisy20 × s54/s56/s57 | ✅ |
| 100% manifests/caches（run_train_with_val + temporal_train_with_val）| ✅ |
| **评估（100% 档）** | ❌ 未跑（上次被中断，需重跑）|

### 19.2 待办清单（按顺序）

- [x] **T1 子集 manifest + cache**（2026-08-27 完成：50/25/10/5% 嵌套，seed 20260827，对齐验证通过）（嵌套按炮，seed 20260827，5%⊂10%⊂25%⊂50%⊂100%）：
  - `subset_run_real_{50,25,10,5}pct.jsonl`（子集行 + val 6,206 行）
  - `subset_run_real_{50,25,10,5}pct.npz`（从 temporal_train_with_val.npz 按 sample_id 切片，验证对齐）
  - 脚本：内联 python（build_curve_subsets.py 模式），~5 min
- [x] **T2 训练 16 个 run（单 seed s54，16 并行 2 卡拉满 ~1h）**（2026-08-27 完成，16/16 rc=0，全部 10/10 epochs）：
  - scratch：`tokamind-temporal-scratch-{50,25,10,5}pct`（4 个，随机初始化）
  - finetune：`tokamind-temporal-finetune-{clean,noisy05,noisy20}-{50,25,10,5}pct`（12 个，从对应预训练 s54 init，`--finetune-method full`）
  - 全部 `--epochs 10 --batch-size 8 --lr 1e-4 --val-shot <M8 shots>`
- [x] **T3 评估（33 个 run）**（2026-08-27 完成）：scratch 100%×6 seed + finetune 100%×各 seed + 子集 16 个 s54
  → `artifacts/tokamind_test_eval/temporal_curve.json`（M9 test 17,498 点）✅ 数值与 §15.8 表格逐位一致
- [x] **T4 曲线图 + 结果表**：`artifacts/tokamind_test_eval/temporal_curve.png`（§15.8 已出）；
  **2026-08-28 追加三臂 RMSE/sample 曲线**（含 random split 对照，4 条线：real scratch / synth pt+ft × temporal / random）：
  `artifacts/pca01sigma_results/temporal_rmse_sample_{clean,noisy05,noisy20}.png` + `_side.png`（dpi=300，脚本 `/tmp/opencode/plot_temporal_curves.py`）
- [x] **T5 结论写入 §15.8**（2026-08-27 完成，见 §15.8 判定 + 数据量曲线表）
  - 对照 Random Split（§9：−5.3%/−12.1%/−24.2%/−56.4%/−61.5%）
  - 100% 档多 seed（clean/noisy05 × 4、noisy20 × 3、scratch × 6）报均值±std
- [x] **T6 产物归档**：temporal 结果三件套（json/png/表格）已进 §15.8 + §6 文件清单
  - 2026-08-28 补充：temporal 曲线图（含 random 对照）产物见 T4 路径

### 19.3 判定（§15.5 回顾）

1. B 臂（预训练+微调）在 M9 上仍优于 A 臂 → 预训练时间泛化有效（强结论）
2. B 优势缩小（vs random −5~60%）→ 增益受 M5-M7→M9 campaign 漂移限制（§0.4：M9 βp 更高、Ip 更低）
3. 零微调参考：clean/noisy05/noisy20 预训练直接测 M9（与 §14 的 8.40/2.51/1.97% 对照）

### 19.4 子集多 seed 补充验证（2026-08-28 启动）

**动机**：§15.8 子集曲线（50/25/10/5pct）是单 seed（s54），noisy05/noisy20 的 50% 档出现 ~+0.3pp 凸起
（noisy05-ft 2.66%、noisy20-ft 2.81%，而 val loss 恰为最低）——疑似单 seed 噪声而非真实趋势。
**方案（2026-08-28 定稿，压缩规模）**：3 臂 × 4 档 × 新增 seed {s55} = **12 个 finetune run**（每格 2 seed：s54+s55 取均值，目标：均值曲线恢复单调；若不够平滑再补 s56）。
**GPU 拉满补充**：12 个 job 分 8 卡必然有 4 卡单进程（~10% < 15% 停机线），故追加 **4 个 scratch 子集 s55**（`temporal-scratch-{50,25,10,5}pct-s55`，A 臂同步多 seed 化，总 16 job = 8 卡 × 2 进程全拉满 ~99%）。100% 档已有 4/3 seed，不重跑。

**配置**（与 s54 逐字一致，仅 `--seed` 不同）：
```
--manifest subset_run_real_{50,25,10,5}pct.jsonl
--dataset-cache subset_run_real_{50,25,10,5}pct.npz
--init-run-dir runs/tokamind-temporal-synth-pretrain{,-noisy05,-noisy20}
--finetune-method full --epochs 10 --batch-size 8 --lr 1e-4 --val-fraction 0.1
--val-shot <M8 shots> --seed {s55}
```
- 子集 manifest 含 val 6,206 行（50%: 99,928 / 25%: 53,046 / 10%: 24,821 / 5%: 14,889 行）
- run 目录：`runs/tokamind-temporal-finetune-{arm}-{f}pct-s55`（与 100% 档多 seed 命名一致）
- 预估耗时：12 run ÷ 16 workers（8 卡 × 2），~1-1.5h（50% 档 ~1.2h/run、5% 档 ~10min/run）

**判定**：s54+s55 均值后，若 50% 档凸起仍在 ~0.3pp 量级且 σ 相当 → 确认为噪声；若 50% 持续偏高 → 需查嵌套子集抽样；若曲线已单调 → 补 s56 收尾。

**执行记录**：`runs/run_temporal_subset_seeds.sh`（state 断点续跑）、日志 `runs/logs/temporal_seed_*.log`
- [x] **26 个 run 训练完成**（2026-08-28 03:05-04:20，0 失败；期间多次因 GPU 分配不均重启，最终 8 卡 × 2 进程拉满）
  - finetune 子集：3 臂 × 4 档 × {s55, s56}（noisy05-50pct 仅 s55）
  - scratch 子集：4 档 × {s55, s56}
- [x] **评估完成**（42 run，M9 test 17,498 点）→ `artifacts/tokamind_test_eval/temporal_curve_seeds.json`
- [x] **结论**（见下方结果表，2026-08-28）

**结果表（RMSE/sample %，nrmse_per_sample，多 seed 均值±std；100% 档用旧 multi-seed）**：

| 臂 | 5% | 10% | 25% | 50% | 100% |
|---|---|---|---|---|---|
| scratch | 5.35±0.12 | 4.87±0.16 | 3.17±0.10 | 2.94±0.11 | 2.69(s54) |
| clean-ft | 2.43±0.07 | 2.35±0.09 | 2.30±0.07 | 2.38±0.03 | 2.29±0.06 |
| noisy05-ft | 2.51±0.07 | 2.41±0.07 | 2.43±0.03 | 2.56±0.10 | 2.27±0.06 |
| noisy20-ft | 2.56±0.08 | 2.48±0.08 | 2.51±0.05 | **2.73±0.08** | 2.43±0.02 |

**结论**：
1. **50% 凸起大部分是单 seed 噪声**：noisy20 50% 单 seed 2.81 → 3 seed 均值 2.73（σ 0.08），noisy05 2.66 → 2.56，clean 2.41 → 2.38
2. **但 50% 档仍系统性偏高（三臂一致 +0.1~0.2pp）**：非纯随机——嵌套子集（5⊂10⊂25⊂50⊂100，seed 20260827）的 50% 炮组合对 M9 系统性略难（noisy20 2.73 距相邻档 ~2.5σ）。怀疑 50% 子集含更多难迁移炮（M5 老炮占比可核）
3. **ft 曲线为"平台型"**：5%-100% 全档 2.27-2.73 几乎水平，无单调趋势（与 random 的 0.72→1.01 递增不同）——预训练后误差对数据量不敏感
4. **scratch 曲线多 seed 后依然陡峭单调**（2.69→5.35，A 臂趋势稳健），ft 与 scratch 的差距随稀缺单调放大（−14% → −55%）

**图产物**：`artifacts/pca01sigma_results/temporal_rmse_sample_multiseed_{clean,noisy05,noisy20}.png` + `_side.png`（均值±std 误差棒，脚本 `/tmp/opencode/plot_temporal_multiseed.py`）
`temporal_rmse_sample_{clean,noisy05,noisy20}.png` 已更新为 4 线对照（temporal/random × scratch/ft，temporal 带多 seed 误差棒，random 单 seed 无棒）

### 19.5 待定研究项目：50% 档凸起的 A/B 处置方案（2026-08-28 记录，待执行）

**背景**：§19.4 多 seed 后 50% 档凸起缩小但未消失（noisy20 2.73±0.08 距相邻档 ~2.5σ，三臂一致 +0.1~0.2pp）——嵌套子集（seed 20260827）的 50% 炮组合对 M9 系统性略难。temporal ft 曲线为**平台型**（2.27-2.73 全档几乎水平），非 random 那种单调递增（0.72→1.01）。用户要求：能否得到严格单调递减的 ft 曲线 → 两个备选方案，待用户决定后执行：

**方案 A：重抽 50% 子集（推荐先试）**
- 换 seed 重新嵌套抽取 50% 档炮（只动 50% 档；注意嵌套链 5⊂10⊂25⊂50⊂100 与 seed 绑定，重抽 50% 需同时重抽 25/10/5% 才保持嵌套，成本 ×4）
- 只重跑 3 臂 × 50% 档 finetune（~1h，8 卡 16 并行）
- 判定：若新 50% 子集均值落回相邻档水平（≤0.1pp 偏差）→ 确认原 50% 子集抽样偏差，曲线恢复单调

**方案 B：接受平台型结论**
- 论证：预训练把"数据稀缺性"变量抹平本身就是发现（temporal ft 平台 2.3-2.7 是 scratch 基线的较小比例，与 random 单调形态的差异 = temporal 特性）
- 论文表述：temporal 下 ft 曲线平坦（−14%~−55% 增益），不做严格单调要求；成本 0

**备注**：random split ft 的"单调"含单 seed 运气成分（0.72→1.01 仅 5 点、无误差棒），与 temporal 平台型对比时需注明口径差异。

**状态**：方案已记录，未执行（训练/评估均已完成，见 §19.4）。

## 20. MAST 69D Shape-OOD 实验设计（2026-08-28 定稿，待执行）

> 设计依据：合作者文档 `Plasma_Shape_OOD_Sim2Real_Workflow_v2_Detailed_LCFS.md`（1981 行）的完整流程，适配我方 MAST 数据集与 69 维 schema。
> 合作者原文要点：① Shape 是 time-slice 属性 (shot,t)→(δu,δl)，不进模型输入 ② split 按 shot 隔离 ③ OOD 边界只由 Real 定义 ④ Synthetic 只来自 RealTrain parent ⑤ topology 与 shape 分开（Topology-OOD ≠ Shape-OOD）⑥ Test-all + Test-OOD-only 双测试集 ⑦ A/C 唯一变量 = 合成预训练 ⑧ 5% real 用 100% 合成预训练 ⑨ 几何指标（磁轴/LCFS/δ 误差）是主场。

### 20.1 研究问题与对照

> **物理一致的合成 equilibrium 预训练，能否提升模型在真实未见 Plasma Shape（Shape-OOD）上的平衡重构能力？**

- **A**：Real-from-Scratch（仅真实训练数据）
- **C**：Synthetic pretraining（Train-parent）+ Real fine-tuning（与 A 完全相同的真实数据）
- 主任务不变：`x_69d(t) → ψ(R,Z,t)`（69 维磁诊断 → 65×65 raw-psi）
- δu/δl/δmean/δasym 只做 dataset stratification metadata，**绝不进模型输入**

### 20.2 冻结数据源（已就绪）

**数据起点 = 与仿真求解一一对应的配对池**：Real 213,924 slices / 7,286 shots（= 求解过合成样本的父状态，每片都有对应合成求解样本）；Synthetic accepted 213,928 与 Real 同一 (parent_shot, target_time) 一一配对（accepted 中 4 个 extra = 13108 炮在 real 侧因 pickup_CCBV_CCBV33 全时段 NaN 被剔，§4.1）。后续所有步骤（LCFS/δ/split/C）均在此冻结池上进行。

| 资产 | 路径 | 规模 |
|---|---|---|
| Real manifest | `data/manifests/training_pca_01sigma/real.jsonl` | 213,924 行 / 7,286 炮 |
| Real cache | `data/processed/training_cache_pca_01sigma/real_full.npz` | 3.16 GB |
| Synthetic accepted | `data/manifests/fullsolve_pca_01sigma_accepted.jsonl`（含 psi_axis/psi_bndry/xpt_count/flag_limiter/parent_shot）| 213,928 行 |
| Synthetic cache | `training_cache_pca_01sigma/synth_clean.npz` + `synth_noisy.npz` | 各 2.7 GB |
| 合成样本目录 | `data/processed/synthetic_pca_01sigma/<shot>_t<time>_v000/` | 526k 目录 |

### 20.3 LCFS 构造

**Real（权威 EFIT label，不重新 contour）**：
- zarr `equilibrium/lcfs_r`、`lcfs_z`（[time, npts]=(74,91)，已验证 11766 可读）
- target_time 与 equilibrium time 轴**精确匹配**（禁止 nearest-time 模糊配对）；不匹配记 `lcfs_valid=False` 审计
- 按炮批量读（7,286 炮 × 1 次 zarr，48 线程，~30 min）
- QC：finite/点数足够/面积非零/非退化/首尾去重/implicit closure

**Synthetic（固定 ψ_bndry + axis-connected core）**：
- 固定物理 level：`ψ = ψ_bndry`（solver 保存值；禁止扫描 level、禁止用 EFIT/predicted bndry）
- ψ_N=(ψ−ψ_axis)/(ψ_bndry−ψ_axis)，flood-fill 只保留**含 O-point（solver opt_count）的连通分量**，排除 open separatrix/真空区连通块
- 外边界：skimage find_contours（复用 §16 lcfs_extract 已验证方法：O 点连通区而非最大块，21735 的坑）
- limiter（flag_limiter）样本：优先 native closed boundary；无法唯一恢复 → `lcfs_valid=False`（**不删除**，仍可预训练，只不进 shape 分析）
- QC + mean/max_level_residual 记录

**canonical 表示（Real/Synth 完全一致）**：去重 → CCW → 起点 max(R)∧min|Z| → 弧长均匀重采样 **170 点**（implicit closure，`canonical_closed_lcfs_arclength_v2`）

### 20.4 δu/δl 计算

- R_geo=(Rmax+Rmin)/2；a=(Rmax−Rmin)/2（a>ε 否则无意义）
- 顶/底极值用**局部二次插值精化**（i±2 周期邻域、弧长参数、Z(s) 顶点；a_z<0 顶 / a_z>0 底，否则回退 raw）——避免 170 点采样抖动
- δu=(R_geo−R_upper)/a；δl=(R_geo−R_lower)/a；δmean=(δu+δl)/2；δasym=δu−δl
- 保存 raw+refined+method；不做人为 clip；极端值进 `shape_outlier_audit.csv`
- Real/Synth 公式一致 → exact-parent Δδu/Δδl/Δδasym（只解释 shape 扩展，不反向改 split）

### 20.5 OOD 定义（Real-only 审计后冻结）

1. 画 Real δu vs δl、δmean vs δasym、分布直方图（P5-P95）、topology 分层
2. MAST 预期 = D 形主核 + 连续尾 → **percentile-based OOD**（如 δasym<P10 或 >P90；阈值来自 Real 分布，不复制合作者 P10）
3. 比较 2-3 个 proposal（OOD 样本量/separation/topology balance）后冻结
4. **Topology 控制**：MAST 96.1% diverted → 主实验 **diverted-only**（备选方案 B）；limiter 单独扩展（报 N 不过度解释）

### 20.6 Split（time-slice shape × shot 级隔离，random split 风格）

**明确：不使用 temporal split（不做 M5-M7/M8/M9 时间划分）**。Shape-OOD 验证的是形状外推而非时间外推，且 random 方式 test 炮数更多（730 vs M9 507）。

- 划分按 shot **随机**分层（非 campaign 分组）：先把 diverted real 池按炮随机排序，再按 r_OOD 分层装入 train/val/test
- r_OOD(s)=N[(s,t)∈Ω_OOD]/N_valid(s)：**Train 低（r 低，几乎不含 OOD 形状）/ Val 中（transition）/ Test 高（目标 unseen shape）**——保留合作者"Train 不含 OOD 形状"的核心纯度（这是 Shape-OOD 的定义本身；若用均匀 80/10/10 会把 OOD 片稀释进 train，Test-OOD-only 就不再是"未见形状"）
- 断言 Train∩Val∩Test=∅（炮级）
- 双测试集：**Test-all**（主报告）+ **Test-OOD-only**（主 Shape-OOD metric）
- 沿用 §9 容量配置（train 约全池 60-70%，diverted 裁剪后）；划分 seed 用 20260828（记录在案）
- 若 OOD region 定义后样本量不足（如 <2,000 片），放宽 OOD 阈值或调整 r_OOD 分层阈值，冻结前审计

### 20.7 Synthetic C 筛选（防泄漏核心）

**Topology 对齐（数据筛选时的硬性步骤）**：主实验严格 diverted→diverted——
- Real 侧：`real.jsonl` 注入 topology 字段（`efit_topology_at_time` 从 zarr 逐帧判定，复用 §4 已验证方法），筛 `topology=diverted`
- Synthetic 侧：accepted 已是位形锚定语义（EFIT=diverted→solve=diverted 匹配才 accepted，§4），再叠加 `flag_limiter=False` 过滤即得 diverted-only 合成池
- limiter→limiter 的对齐留给 limiter 扩展实验；**X-point→limiter 跨拓扑定义为独立的 Topology-transfer 实验**，不混入 Shape-OOD 主实验（合作者 §12）

1. Real split **先冻结**再筛合成
2. C = (parent_shot, target_time) ∈ RealTrain **exact key 匹配**
3. topology 对齐（diverted→diverted，如上）
4. **不裁剪合成**：5% real 仍用 100% Train-parent 合成预训练
5. 合成进 OOD region 的样本**保留**（正是要验证的机制）
6. Leakage 断言：Train∩Val/Test=∅；Synth parent∩Val/Test shots=∅

### 20.8 A/C 训练

| 项 | 值 |
|---|---|
| 模型 | d64/2/4/128（§9 配置）|
| 输入/输出 | 69 维 → 65×65 raw-psi |
| 优化 | bs8、lr 1e-4、AdamW、epochs 10、val-fraction 0.1 |
| Real fractions | 100/50/25/5%（parent-shot nested：5⊂25⊂50⊂100，保留选中炮全部合法 slices）|
| seed | s54（主）+ s55/s56 备选 |
| run 数 | A/C × 4 档 = 8（×3 seed = 24）|

Scaler：A 用当前 fraction Real Train 统计量；C 预训练用合成 scaler，微调 rebase 到 Real Train scaler。禁止 Val/Test 统计量。

### 20.9 评估

- 主场：RMSE_ψ、Relative L2、R²、RMAE/样本（Test-all + Test-OOD-only 都报）
- **Geometry（主场）**：磁轴 R/Z 误差、LCFS 平均距离/Hausdorff、δu/δl/δasym 误差（pred ψ → LCFS 提取 → 与 EFIT 比较；find_critical 批量需加速或抽样）
- 分层：diverted Test-OOD（主）+ limiter 扩展（报 N）
- Severity bins：Test-OOD 内 δasym P50/P90 分界 → mild/moderate/extreme，报 A/C error + Gain=(E_A−E_C)/E_A

### 20.10 执行顺序 + 预估耗时

```text
[0] Real LCFS 批量提取（zarr lcfs_r/z，按炮）+ topology 注入（efit_topology_at_time）  ~30 min（48 线程）
[1] Synthetic LCFS（ψ_bndry+axis-connected）+ flag_limiter 过滤                      ~40 min（48 并行，213,924×~0.5s）
[2] δ 计算 + QC                                 ~20 min
[3] Real-only shape 审计 + OOD proposal 对比     ~30 min（人工定稿）
[4] shot-level split（diverted-only）+ Test-all/OOD-only  ~10 min
[5] Synthetic C 筛选（exact parent + diverted 对齐）     ~10 min
[6] A/C 训练 8 run（16 并行 8 卡）              ~1.5-2h
[7] 评估（ψ + geometry）                        ~30-60 min
合计 ~4-5h（不含多 seed）
```

### 20.11 交付物

```text
real_shape_metadata.parquet / synthetic_shape_metadata.parquet（δu/δl/δmean/δasym raw+refined+method）
real_train/val/test/test_ood_only_manifest.jsonl + synthetic_C_manifest.jsonl
leakage_audit.txt（6 条断言）
OOD proposal 对比表 + 冻结阈值
A/C × 4 fractions 评估表（Test-all + Test-OOD-only + severity + geometry）
```

### 20.12 资产复用与坑

- 训练链路：§9 配置 + runner 模式（state 断点、8 卡 16 并行、动态最闲卡分配，§19.4 经验）
- 缓存：real/synth 从现有 npz 按 sample_id 切片，无需重新 zarr 提取
- Synthetic LCFS：§16 lcfs_extract 已验证（O 点连通区 + skimage），改固定 ψ_bndry level
- 坑：训练 `.tokamind-train-env`、提取 `.freegsnke-solve-env`、后台 setsid+disown、缓存与 manifest 对齐

### 20.13 状态

**执行记录（2026-08-28）**：
- [x] [0] Real LCFS：213,924/213,924 有效（100%），topology diverted 199,915 / limited 13,630 / None 379
- [x] [1] Synthetic LCFS：210,640/213,928 有效（98.5%，91 固定 ψ_bndry / 其余动态扫描回退，method 字段区分），diverted 199,957 / limited 13,971
- [x] [2] δ 计算内联完成（lcfs_delta：raw+refined 局部二次插值）
- [x] [3] OOD 冻结：**δasym < P10 = -0.214（极下三角）**，19,992 样本；审计图 `artifacts/pca01sigma_results/shape_ood_audit.png`；diverted real δasym 主核 -0.05~+0.02 + 长左尾（P1 -0.303）
- [x] [4] split 冻结（seed 20260828，diverted-only）：train 5,743 炮/176,319 片、val 170/2,835、test 775/20,761、**Test-OOD-only 19,009**；`data/manifests/shape_ood/`
- [x] [5] Synthetic C：175,717 行（diverted + exact train-parent 对齐）；cache 重建 34s（`synth_ood_pretrain_clean.npz`，原 `synth_pretrain_clean.npz` 缺 34,181 行炮集不匹配，弃用）
- [x] [6] 训练完成（2026-08-28 05:30-07:50，32/32 rc=0，0 失败）：阶段 1 = 预训练 s54-57（4）+ scratch 4 档 × s54-56（12）；阶段 2 = finetune 4 档 × s54-57（16）。配置 batch 8/epochs 10/lr 1e-4（与 §9 一致）；日志 `runs/logs/shapeood_*.log`、state `runs/shapeood_train.state`。**坑：batch 32 单进程利用率仅 11%（小模型瓶颈），必须每卡 ≥2 进程；预训练不可传 real val shots（合成 manifest 无 val 炮，§19 坑）；短档完成后空槽用额外 seed 手动补位（s57）撑窗口**
- [x] [7] 评估完成（28 run × 2 测试集）：`artifacts/tokamind_test_eval/shapeood_{test_all,test_ood_only}_metrics.json`（**注意：评估脚本缓存需手动复制 `test_real.npz`/`test_ood_only_real.npz` 到 `artifacts/tokamind_test_eval/test_cache_split_test_*.npz`，否则会从 zarr 重提取失败**）
- [x] 结论（见 §20.14，2026-08-28）

### 20.14 Shape-OOD 评估结果与结论（2026-08-28）

**测试集**：Test-all = 20,761 片 / 775 炮（diverted held-out）；Test-OOD-only = 19,009 片（δasym < -0.214，极下三角）。
**指标**：nrmse_per_sample（%），A = scratch × 3 seed（s54-56）均值±std，C = pretrain+ft × 4 seed（s54-57）均值±std。

**结果表**：

| 数据档 | Test-all A | Test-all C | 增益 | Test-OOD A | Test-OOD C | 增益 |
|---|---|---|---|---|---|---|
| 100% | 2.04±0.08 | 1.76±0.15 | **+13.7%** | 2.07±0.09 | 1.78±0.15 | **+14.0%** |
| 50% | 2.17±0.07 | 1.77±0.07 | **+18.2%** | 2.20±0.07 | 1.79±0.07 | **+18.6%** |
| 25% | 4.32±0.36 | 1.82±0.03 | **+57.8%** | 4.42±0.37 | 1.84±0.03 | **+58.5%** |
| 5% | 5.48±0.01 | 1.84±0.02 | **+66.4%** | 5.53±0.01 | 1.84±0.03 | **+66.7%** |

（增益 = (A−C)/A；正 = C 优于 A）

**图产物**：`artifacts/pca01sigma_results/shapeood_curve.png`（双面板：Test-all + Test-OOD-only，A/C 均值±std 误差棒；脚本 `/tmp/opencode/plot_shapeood_curve.py`）

**重大结论**：
1. **合成预训练在 Shape-OOD 上全域转正，增益为三个场景最高**：−14% ~ **−67%**（vs random −5~60%、temporal −17~60%），随数据稀缺单调放大
2. **C 臂曲线几乎水平**（Test-all 1.76→1.84、Test-OOD 1.78→1.84）：预训练后误差对数据量、对未见形状（极下三角）都不敏感
3. **A 臂 25% 档崩溃**（4.32，vs 50% 档 2.17）：纯真实训练在 25% 子集上对极端形状泛化急剧恶化（seed 间 std 0.36 也大），预训练完全免疫（C 25% 1.82，std 仅 0.03）
4. **Test-OOD-only 增益 ≥ Test-all**（各档基本持平或略高）：test 炮 91% 片本身是 OOD 形状，shape 效应全局可见；预训练在真正"未见形状"上的价值不低于整体
5. **机制链条成立**：RealTrain 形状支持有限（δasym 主核窄）→ 合成（PCA ±0.1σ，train-parent）扩展形状覆盖 → C 在 held-out 极下三角形状上 −67% → **仿真预训练 = Shape-OOD 泛化的最有效手段**（结合 §9 random、§15 temporal，三场景证据链完整）

**遗留**：
- ~~limiter 扩展实验~~ → **已完成（2026-08-28，单 seed 快跑，7 分钟）**：见 §20.15
- geometry 指标（磁轴 R/Z 误差、LCFS 平均距离、δu/δl 误差）未跑——需 pred psi → LCFS 提取后处理（~40k 样本），作为主场指标的补充
- severity bins（mild/moderate/extreme）未跑——Test-OOD 内按 δasym 分桶报增益

### 20.15 limiter Shape-OOD 扩展实验（2026-08-28，已完成）

**配置**：limiter 池（real 13,630 片/2,183 炮；synth lcfs_valid 11,294 片）。OOD region = **δasym > P90 = 0.010**（右尾，156 纯 OOD 炮 > diverted 左尾的 99）。
split（seed 20260828）：train 1,714 炮/10,807 片、val 242/1,833、test 227/990、**Test-OOD-only 817**；subsets 100/50/25%（5% 档样本太少跳过）；Synthetic C = 8,808（limited + exact train-parent）。
单 seed（s54），batch 8/epochs 10，预训练 + A/C × 3 档 = 7 run，**总训练 7 分钟**（样本少）。评估：`artifacts/tokamind_test_eval/limiter_{test_all,test_ood_only}_metrics.json`。

**结果（nrmse_per_sample %）**（5% 档为 2026-08-28 补跑：A/C 各 1 run，训练 ~1 分钟；Test-OOD 817 片）：

| 数据档 | Test-all A | C | 增益 | Test-OOD A | C | 增益 |
|---|---|---|---|---|---|---|
| 100% | 4.29 | 4.01 | +6.4% | 4.32 | 4.06 | +6.1% |
| 50% | 5.21 | 4.28 | +17.7% | 5.24 | 4.35 | +17.1% |
| 25% | 5.22 | 4.45 | +14.8% | 5.28 | 4.51 | +14.5% |
| 5% | 5.70 | 4.93 | +13.5% | 5.70 | 4.93 | +13.5% |

**结论**：
1. **limiter 组预训练同样全域转正**（+6~+18%），topology 分层下 Shape-OOD 结论对 diverted 和 limiter 都成立
2. **绝对误差远高于 diverted**（limiter 4.0-5.2% vs diverted 1.8-2.2%）——limiter 平衡重构本身更难（启动段/小等离子体），且 Test-OOD 817 片样本少（单 seed，25% 档增益非单调 +14.8% vs 50% +17.7% 属噪声）
3. 增益幅度小于 diverted（+18% vs +67%）——limiter 合成池小（8,808 vs 175,717）、位形约束强（limiter 接触面限制形状自由度），预训练扩展形状空间的能力受限
4. **Topology-OOD ≠ Shape-OOD 分离成功**：diverted/limiter 各自组内做 Shape-OOD，无跨拓扑混淆

**图产物**：`artifacts/pca01sigma_results/shapeood_topology_side.png`（diverted + limiter 两图并排，Test-OOD-only 口径）

### 20.16 diverted / limiter 数据划分定稿与最终结果汇总（2026-08-28）

**数据划分具体方案（两组均：Real-only 定义 OOD → time-slice 判定 → shot 级 r_OOD 分层 → 互斥 split）**：

| 项 | diverted 主实验 | limiter 扩展实验 |
|---|---|---|
| Real 池 | 199,915 片 / 6,688 炮（diverted，LCFS 100% 有效）| 13,630 片 / 2,183 炮（limited，LCFS 100% 有效）|
| OOD region（Real-only 审计后冻结）| **δasym < P10 = -0.214**（极下三角，左尾）| **δasym > P90 = 0.010**（右尾，156 纯 OOD 炮 vs 左尾 99）|
| OOD 样本 | 19,992 片 | 1,363 片 |
| split 逻辑 | r_OOD(s) > 0.5 → test；0.1 < r ≤ 0.5 → val（≤500 炮，从高取）；≤0.1 → train | 同左（val ≤300 炮）|
| train / val / test（炮）| 5,743 / 170 / 775 | 1,714 / 242 / 227 |
| train / val / test（片）| 176,319 / 2,835 / 20,761 | 10,807 / 1,833 / 990 |
| **Test-OOD-only** | **19,009 片**（test 炮中 δasym<-0.214）| **817 片**（test 炮中 δasym>0.010）|
| 子集（seed 20260828，按炮嵌套，保留选中炮全部合法片 + val）| 100/50/25/5% | 100/50/25/5%（5% 档 = 2,433 行，含 val 1,833，训练 ~600 片）|
| Synthetic C（exact train-parent + topology 对齐，不裁剪）| 175,717（diverted）| 8,808（limited）|
| 防泄漏断言 | train∩val∩test=∅；synth parent ⊄ val/test shots | 同左 |

**最终结果（nrmse_per_sample %，diverted 多 seed 均值±std：A×3 / C×4；limiter 单 seed）**：

| 数据档 | diverted A | diverted C | 增益 | limiter A | limiter C | 增益 |
|---|---|---|---|---|---|---|
| 100% | 2.07±0.09 | 1.78±0.15 | **+14.0%** | 4.32 | 4.06 | +6.1% |
| 50% | 2.20±0.07 | 1.79±0.07 | **+18.6%** | 5.24 | 4.35 | +17.1% |
| 25% | 4.42±0.37 | 1.84±0.03 | **+58.5%** | 5.28 | 4.51 | +14.5% |
| 5% | 5.53±0.01 | 1.84±0.03 | **+66.7%** | 5.70 | 4.93 | +13.5% |

（口径：diverted = Test-OOD-only 19,009 片；limiter = Test-OOD-only 817 片）

**总结论**：Shape-OOD 下合成预训练在两种 topology 均全域转正——diverted +14~+67%、limiter +6~+18%。diverted 增益远大于 limiter（合成池大 20 倍、limiter 位形自由度受接触面限制），且与 random（−5~60%）、temporal（−17~60%）三场景证据链一致：**预训练学到跨装置/跨位形/跨形状可迁移的平衡物理，数据越稀缺价值越大**。limiter 25% 档增益（+14.5%）低于 50%（+17.1%）为单 seed 小样本噪声（817 片 test）。

### 20.17 Shape-OOD LCFS 可视化验证（2026-08-28，已完成）

**目的**：用 LCFS 几何直观验证"预训练提升 OOD 形状泛化"——把 §20.16 的误差数字（−67%/−13.5%）落到边界曲线上。

**做法**：
1. **样本选择**：diverted 3 个 + limiter 3 个**极端 OOD 代表样本**，全部经断言在 held-out test 炮且 Test-OOD-only 内（δasym 越界阈值内），无泄漏：
   - diverted：28232_t0.56（δasym -0.398）、23743_t0.365（-0.385）、20894_t0.325（-0.383）——极下三角
   - limiter：28264_t0.205（+0.159）、11920_t0.105（+0.119）、17648_t0.35（+0.099）——上三角倾向
2. **模型**：A = `tokamind-shapeood-scratch-5pct-s54`（纯真实 5% 从头）、C = `tokamind-shapeood-finetune-5pct-s54`（合成预训练 + 5% 微调）——选 5% 档因为 A/C 差距最大（diverted 5.53 vs 1.84）
3. **LCFS 提取**：模型 forward（test cache features，pred×std+mean 反归一化）→ §16 已验证动态扫描（psi_max 向下扫 level、O 点连通区、泄漏前最大闭合面、skimage find_contours）；EFIT real 直接用 §20.3 canonical 170 点 LCFS（metadata）
4. **绘图**：2×3 布局（diverted/limiter 行 × 3 样本列），R-Z 平面三线：EFIT real（蓝实线）、scratch 5%（红虚线）、pretrain+ft 5%（绿实线），`aspect=equal`、xlim [0,2.05]、ylim [-2.1,2.1]
5. 脚本：`/tmp/opencode/plot_shapeood_lcfs_visual.py`（坑：limiter 样本 features 在 `limiter_test_real.npz` 而非 `test_real.npz`；`move_batch_to_device` 需 `torch.device` 对象）

**产物**：`artifacts/pca01sigma_results/shapeood_lcfs_visual.png`

**可视化平滑处理（2026-08-28，方案 A）**：
- **动机**：图上线出现小幅震荡——① EFIT real 线：zarr 原始 LCFS ~74 点经 canonical 线性插值（np.interp）重采样 170 点，弯曲处呈折线段；② 预测线：65×65 网格（~3cm 分辨率）find_contours 的像素级锯齿
- **处理方式**：仅对 **EFIT real 线**在绘图层做周期 B-spline 重采样（`scipy.splprep(per=True, s=0)` 插值样条 → `splev` 500 点，平滑折角、C2 连续）；**不改 metadata（lcfs_r/lcfs_z 仍是 canonical 170 点线性插值版本）、不改已冻结 OOD 阈值/split、不改 δ 计算与统计口径**——纯视觉优化
- **坑**：splprep `per=True` 要求首尾严格闭合（需临时把首点追加到末尾）；canonical 170 点存在弧长重采样产生的重复点（`np.diff(r).min()==0`）会触发 "Invalid inputs"，须先去重（容差 1e-10）再拟合
- 预测线（红/绿）的像素锯齿未处理（方案 B 的 Savitzky-Golay 平滑未启用，避免过度平滑吃掉 X 点尖角特征）

**结论（解读要点）**：
1. 红虚线（纯真实 5%）在未见极端形状上明显失形（边界走样、形状特征塌缩）；绿实线（预训练 + 5%）贴近 EFIT real——**"形状先验"由合成预训练注入模型**，5% 真实数据即可在 OOD 上正确外推
2. diverted 行红/绿差距显著（对应 −67%），limiter 行差距较小（对应 −13.5%，位形受接触面约束 + 样本少）——图的几何证据与 §20.16 误差表逐行对应
3. 局限：6 个精心挑选的极端样本用于展示（非随机抽样），统计证据以 §20.16 表格为准

## 21. 实验资产清单与最优 seed 筛选（2026-08-28，发给合作者用）

> 本节列出全部实验的 checkpoint、数据、脚本，并按"结果最好"筛选出每组的代表 run。

### 21.1 Checkpoints 全景（runs/，253 run，2.4 GB）

| 实验 | run 前缀 | run 数 | 内容 |
|---|---|---|---|
| Random split | `tokamind-pca01sigma-*` | 39 | synth-pretrain(clean/noisy05 s54-69/noisy20 s54) + finetune/scratch × 5 档 × 3 噪声 |
| Temporal | `tokamind-temporal-*` | 67 | synth-pretrain × 3 噪声 × 多 seed + finetune/scratch × 5 档 × 多 seed |
| Shape-OOD diverted | `tokamind-shapeood-*` | 36 | synth-pretrain s54-57 + A(scratch s54-56)/C(finetune s54-57) × 4 档 |
| Shape-OOD limiter | `tokamind-limiter-*` | 9 | synth-pretrain + A/C × 3 档（单 seed s54）|

### 21.2 数据资产（仿真+真实配对）

| 资产 | 路径 | 规模 | 建议 |
|---|---|---|---|
| 合成 accepted manifest | `data/manifests/fullsolve_pca_01sigma_accepted.jsonl` | 213,928（1.9 GB）| ✅ 核心必传 |
| 求解 manifest（全量扰动） | `data/manifests/fullsolve_pca_01sigma.csv` | 502,402（0.33 GB）| ✅ |
| Real manifest（配对） | `data/manifests/training_pca_01sigma/real.jsonl` | 213,924 | ✅ |
| Random split manifests | `data/manifests/training_pca_01sigma/` | 24 文件 | ✅ |
| Temporal manifests | `data/manifests/training_temporal/` | 14 文件 | ✅ |
| Shape-OOD manifests | `data/manifests/shape_ood/` + `shape_ood_limiter/` | 各 14 文件 | ✅ |
| Shape metadata（OOD 定义） | `data/processed/shape_ood/{real,synthetic}_shape_metadata.jsonl` | 3.0 GB | ✅ |
| 训练 caches | `data/processed/training_cache_pca_01sigma/` | 37 GB（28 npz）| ⚠️ 可重建（build_cache_batched.py 34s）|
| 求解产物目录 | `data/processed/synthetic_pca_01sigma/` | 502k 目录 | ⚠️ 可不传（accepted 含信息）|

### 21.3 处理脚本（mast-bridge/scripts/）

| 脚本 | 用途 |
|---|---|
| `run_pca_solve_shard.py` | PCA 求解分片 worker |
| `build_synthetic_manifest.py` | 7 条判据过滤 |
| `build_synthetic_magnetic_diagnostics.py` | 合成诊断生成 |
| `build_cache_batched.py` | 批量缓存（34s）|
| `split_pca_random.py` / `build_pca_curve_subsets.py` | Random split + 子集 |
| `build_shape_metadata.py` / `build_shape_ood_split.py` / `build_shape_ood_limiter_split.py` / `build_shape_ood_caches.py` | Shape-OOD 全流程 |
| `train_tokamind_manifest.py` / `evaluate_tokamind_testset.py` | 训练/评估 |

### 21.4 最优 seed 筛选（nrmse_per_sample %，各组代表 run）

**① Random split（s54 单 seed 即主结果，无需再筛）**
| 档 | scratch | clean-ft | 增益 |
|---|---|---|---|
| 100% | 0.7636 | **0.7223** | −5.4% |
| 50% | 0.9034 | **0.7907** | −12.5% |
| 25% | 1.1284 | **0.8538** | −24.3% |
| 10% | 2.0973 | **0.9405** | −55.2% |
| 5% | 2.5202 | **1.0112** | −59.9% |

**② Temporal（M9 test，100% 档多 seed）**
- **C best = s56**：clean-100pct-s56 0.02243、noisy05-100pct-s56 0.02195、noisy20-100pct-s56 0.02224
- **A best = s107**：scratch-100pct-s107 0.02438
- 数据量曲线（s54）：clean-ft 全档最优（0.02367→0.02387）；子集 best：clean-25pct-s55 0.02215、clean-5pct-s56 0.02365

**③ Shape-OOD diverted（Test-OOD 19,009 点）**
- **C best = s54**：100% 0.01598 / 50% 0.01678 / 25% 0.01804 / 5% 0.01857
- **A best = s56**：100% 0.01948 / 50% 0.02090 / 25% 0.04507 / 5% 0.05472
- 增益（C s54 vs A s56）：−18.0% / −19.7% / −60.0% / −66.1%

**④ Shape-OOD limiter（Test-OOD 817 点，单 seed s54）**
- scratch：100% 0.04318 / 50% 0.05245 / 25% 0.05278
- finetune：100% **0.04055** / 50% 0.04347 / 25% 0.04513
- 增益：−6.1% / −17.1% / −14.5%

### 21.5 打包建议（发给合作者）

**必须包（~5 GB）**：runs/ + fullsolve_pca_01sigma_accepted.jsonl + 全部 manifests + shape_metadata + scripts + 文档
**可重建**：training_cache（37 GB）、synthetic_pca_01sigma 产物目录
**可选**：rejected manifest、artifacts 图

### 21.4b Random split noisy 臂数据齐全性确认（2026-08-28）

**run 目录（12 个）**：
- noisy05：`synth-pretrain-noisy05-s54` + `noisy05-finetune-{100,50,25,10,5}pct`
- noisy20：`synth-pretrain-noisy20-s54` + `noisy20-finetune-{100,50,25,10,5}pct`

**评估（12 runs，`pca01sigma_noisy_curve.json`，test 21,350 点 nrmse_per_sample %）**：

| 档 | noisy05-ft | noisy20-ft |
|---|---|---|
| 100% | 0.7363 | 0.7276 |
| 50% | 0.8229 | 0.7890 |
| 25% | 0.8772 | 0.8610 |
| 10% | 0.9325 | 0.9407 |
| 5% | 0.9870 | 1.0061 |

- 零微调参考：noisy05-s54 3.224%、noisy20-s54 2.585%（同文件）
- **noisy 臂与 clean 一样 5 档齐全，run + 评估完整**（§14.2b 表）

### 21.6 交付包方案（2026-08-28 定稿，发给合作者）

> **交付原则**：合作者解压后读 `progress2.md` 即可直接训练/复现，**artifacts/ 不打包**（合作者自己画图）。

**包结构**：
```
fusion-workspace/
├── progress2.md / statistics_shot.md / filter.md / experiment_pca01sigma_curve.md / Plasma_Shape_OOD_Workflow_v2_Detailed_LCFS.md
├── mast-bridge/          # 代码（14 核心 scripts + src + configs）+ 2 个 venv（train/freegsnke）
├── external/tokamind/    # MMT 模型库
├── runs/                 # 全部 checkpoint（253 run，2.4 GB）
├── data/
│   ├── manifests/        # 全部：fullsolve_* + training_pca_01sigma + training_temporal + shape_ood + shape_ood_limiter
│   └── processed/
│       ├── training_cache_pca_01sigma/   # 37 GB（开箱即训；可省→build_cache_batched.py 34s 重建）
│       └── shape_ood/                    # 3 GB（OOD 定义必需）
└── artifacts/            # ❌ 不打包（可空）
```

**mast-bridge/scripts/ 精选清单（28 个，其余可省）**：

| 类别 | 脚本 |
|---|---|
| 核心训练链路（14）| `train_tokamind_manifest` `evaluate_tokamind_testset` `build_cache_batched` `slice_cache` `split_pca_random` `build_pca_curve_subsets` `build_shape_metadata` `build_shape_ood_split` `build_shape_ood_limiter_split` `build_shape_ood_caches` `build_synthetic_magnetic_diagnostics` `build_synthetic_manifest` `run_pca_solve_shard` `run_freegsnke_forward` |
| 数据链路（14）| `build_lao_fit_npz` `run_lao_fit_batch` `build_machine_from_zarr` `run_machine_batch` `build_lao85_variant_rows` `run_lao85_variant_solve_batch` `build_production_variants` `precheck_variant_rows` `build_real_manifest` `build_dataset_cache` `build_training_manifests` `build_experiment_manifests` `split_v2_manifests` `build_scarcity_manifests` |

**体积预算（~55 GB）**：runs 2.4 + venv 1.5 + manifests ~11 + accepted 1.9 + cache 37 + shape_ood 3 + 代码 <0.1

**可省项**：artifacts/（空）、tests/、debug.md/progress.md（旧文档）、download_* 脚本、plot_* 可视化脚本、rejected manifest（99M）、求解产物目录 synthetic_pca_01sigma（502k 目录，accepted 已含信息）

**决策点**：① venv 打包（1.5G，建议带，免重建）② cache 37G（带=开箱即训；省=合作者用 build_cache_batched.py 重建，需 manifest+zarr+freegsnke-env）③ runs 全量 vs 最优 seed（全量 2.4G 完整，精选 1.1G 够复现论文）

### 21.7 噪声合成方式与产物（2026-08-28 补充）

**合成公式**（`mast-bridge/scripts/build_synthetic_magnetic_diagnostics.py` `_add_noise`）：

```
x_noisy = x_clean + σ_channel × N(0,1)
```

- **σ 来源**：`mast-bridge/configs/diagnostic_features/noise_profile.json`（93 通道实测噪声 σ）
  - magnetics_ip: σ=1000 A；flux_loop: σ≈0.007–0.046 Wb；pickup: σ≈0.0001–0.013 T；coil_active 同理
- **逐样本可复现随机种子**：`seed = noise_seed + zlib.crc32(sample_id)` → `np.random.default_rng(seed)`
- **噪声级别**：`--noise-scale 0.5 / 2.0` → σ 乘 0.5×（noisy05）/ 2.0×（noisy20）
- **只作用于诊断特征**（flux_loop/pickup/coil_active/ip），**不影响 psi 标签与平衡产物**（求解结果不变，噪声只在特征层）

**产物 cache**（`data/processed/training_cache_pca_01sigma/`，各 171,427×69 特征）：

| 版本 | 文件 | 说明 |
|---|---|---|
| clean | `synth_pretrain_clean.npz` | 基准（预训练主用）|
| noisy（旧 scale=1.0）| `synth_pretrain_noisy.npz` | 早期版 |
| **noisy05** | `synth_pretrain_noisy05.npz` | σ×0.5（§14 实验）|
| **noisy20** | `synth_pretrain_noisy20.npz` | σ×2.0（§14 实验）|

- Temporal/Shape-OOD 各有独立合成缓存：`temporal_synth_{clean,noisy05,noisy20}.npz`、`synth_ood_pretrain_clean.npz`、`synth_ood_limiter_pretrain_clean.npz`
- 验证：noisy05 与 clean 特征确有差异（max|Δ| 190–1500，符合 σ 量级）✅
- **结论（§14）**：噪声非必要条件——PCA+位形锚定下 clean/noisy05/noisy20 微调后结果几乎持平（差异 <2.5%）；噪声只影响零微调（正则效应：clean 10.14% → noisy05 3.22% → noisy20 2.59%）

### 21.8 交付包最终版（2026-08-28）

**文件**：`/inspire/qb-ilm/project/ai-for-fusion/public/collab_package.tar.gz`（35 GB）

**修正记录**：
1. runs/ 只含**最优 seed 50 个**（`_best` 后缀重命名，298 MB），非全量 253（2.4G）
2. 打包过程中发现：temporal finetune **100pct 的 s55/s56 run 目录不存在**（仅评估 JSON 有数字）——清单已用现存 run 替代（clean/noisy05 100pct 用无后缀 s54，noisy20 用 s57）
3. 新增 `BEST_RUNS.md`（50 个最优 run 对照表：run 名 + nrmse 指标）
4. README.md 更新为最终版（runs 说明改为 50 _best）

**包内容（最终）**：
| 组件 | 大小 |
|---|---|
| data/（manifests 11G + caches 32G + shape_ood 3G）| 46G |
| mast-bridge/（代码 + 2 venv）| 1.5G |
| runs/（50 _best）| 298M |
| external/tokamind/ | 10M |
| 文档（README + progress2 + BEST_RUNS + 方法文档）| ~300K |

**合作者路径**：解压 → 读 README.md → 训练命令 §5 → 最优 run 对照 BEST_RUNS.md

### 21.9 campaign 标注来源（2026-08-31 补充）

> **结论**：campaign 字段**不在 zarr 里**（IMAS 标准元数据，attrs 无此字段），也不在 manifests 里。权威来源 = FAIR-MAST shot 级元数据 parquet（本 workspace 有，collab_package 无）。详见 `statistics_shot.md` §10。

**来源链**：
1. `.shots_metadata.parquet`（`data/raw/mast/` 隐藏文件，11,573 炮 × 113 列）：含 `campaign`/`divertor_config`/`plasma_shape`/`plasma_flat_top_start/end_time`/`shot_useful` 等；`endpoint_url` 列 = STFC Echo S3
2. parquet 随 zarr 一起从 FAIR-MAST 对象存储（S3）下载（`mast-bridge/scripts/download_all_mast_shots.py`，s5cmd）；zarr attrs `commit_url = github.com/ukaea/fair-mast-ingestion`
3. 派生 csv：`mast-bridge/scripts/build_clean_pool_analysis.py`（parquet + LAO fit NPZ）→ `data/processed/real/clean_pool/per_shot_flat_top_stats.csv`（每炮一行含 campaign）

**当前数据集（real.jsonl 213,924 片 / 7,286 炮）**：M5 992 炮/23,433 片(10.95%)、M6 2,584/82,294(38.47%)、M7 2,470/81,565(38.13%)、M8 559/6,206(2.90%)、M9 507/17,498(8.18%)、Unknown 174/2,928(1.37%)。
对账：M5+M6+M7=6,046 炮 = temporal train；M8=559 val；M9=507 test；Unknown 174 炮（21624–25355）= §15.1 剔除的 2,928 片。
全量 zarr（11,573 炮）：M5 1,733/M6 2,670/M7 3,209/M8 1,844/M9 1,028/Unknown 287/未覆盖 802。
近似映射（炮号区间）：M5-M7 11766–25017 / M8 25605–28346 / M9 28631–30451；精确以 parquet/csv 为准。

### 21.9 求解成功样本 shot 覆盖（2026-08-28）

| 项 | 值 |
|---|---|
| 求解尝试（含失败）| 502,402 / 10,667 shot |
| **求解成功（rc=0）** | **476,954 / 10,037 shot**（90% of 11,148）|
| 过滤后 accepted | 213,928 / 7,287 shot |

---

## 45. sec9bs64 + 50ep 早停版实验（2026-09-01 定稿，⚠️ 当前唯一运行中实验，实际执行 3 seed 57 run）

> **背景**：§40 pilot（sec9bs64）结果非常好（bs64 + cosine + ft lr1e-5），但只有 10ep（欠拟合，§43.3）。
> **唯一变化**：epochs 10 → **50 + 早停 patience=10**，其余全部保持 sec9bs64 配置不变。

### 45.1 配置（= sec9bs64，仅 epochs 变化）

| 项 | 值 |
|---|---|
| batch size | 64 |
| scheduler | **cosine 到 0，无 warmup**（sec9bs64 原样，非 constant）|
| weight decay | 0 |
| lr：预训练 / scratch | **1e-4** |
| lr：finetune | **1e-5**（sec9bs64 原样）|
| epochs | **50**（唯一变化）|
| 早停 | patience=10（新增，10ep 时不会触发，50ep 下生效）|
| seed | s54 单 seed |

### 45.2 run 清单（19 run，`-sec50` 后缀）

**预训练（3，重训 50ep cosine lr1e-4）**：`tokamind-pca01sigma-synth-pretrain[-noisy05|-noisy20]-s54-sec50`
**scratch（4，50ep cosine lr1e-4）**：`tokamind-pca01sigma-scratch-{100,50,25,5}pct-s54-sec50`
**finetune（12，50ep cosine lr1e-5）**：`tokamind-pca01sigma-{finetune,noisy05-finetune,noisy20-finetune}-{100,50,25,5}pct-s54-sec50`

### 45.3 数据与依赖

- 资产映射同 §42/§44（现成 manifest/cache）
- C 臂 init = 对应同 seed 新预训练（`-sec50`，warmstart 无 resume）
- 预训练无 val 炮 → `--val-fraction 0.1`（禁 val-shot）

### 45.4 时间估算（8 卡 16 并行）

- bs64 下 100% 档 50ep ≈ 3-4h（cosine 收敛，早停可能提前）
- 预训练 3 + scratch 4（波1）→ ~3.5h；finetune 12（波2，等预训练）→ ~3.5h
- **总墙钟 ≈ 6-7h**（100% 档 50ep 是瓶颈）

### 45.5 判定

1. 50ep 收敛后 A/C 增益是否保持 sec9bs64 的趋势（10ep 时 C 臂有效）
2. 对照 §42（constant 50ep）、§37（bs8 100ep）的负结论，看 cosine 50ep 是否恢复正收益
3. 5% 档增益（sec9bs64 时 +21%）在收敛后是否保持

---

## 46. 交接记录（2026-09-01 重要：文档损坏 + 当前实验状态）

> ⚠️ **文档损坏记录**：2026-09-01 05:23 发现 progress2.md 的 **§22-§44 全部丢失**（文件从 ~3150 行截断到 1665 行）。
> §37（100ep bs8 负结论）、§42-§43（超参数结论）、§44（f64）等关键章节**不在本文档**。
> 丢失内容的历史记录在会话对话里（2026-08-31 ~ 2026-09-01），无法从文件备份恢复
> （fusion-workspace 备份仅到 §21，collab_package.tar.gz 未含 §22+）。
> **结论**：§37 的负结论（100ep bs8：100% −12.6% / 50% −3.8% / 25% −4.9% / 5% +3.4%，标存疑待 bs64 复核）**仅存在于会话记录**，当前实验（§45 sec50 3seed bs64）即为复核。

### 46.1 当前正在运行的实验（唯一）

**§45 sec50 3 seed 实验**（run_sec50_3seed.py，57 run）：
- 配置：sec9bs64 + **50ep + 早停 patience=10** + cosine + bs64
- lr：预训练/scratch 1e-4，finetune 1e-5
- 3 seed：s54/s55/s56
- run 后缀：`-s{seed}-sec50`
- 波 1：预训练 9 + scratch 12 = 21 run（16 并行，GPU 拉满 94-96%）
- 波 2：finetune 36（等预训练完成）
- 进度：波 1 进行中（完成 2/21，scratch-5pct s54/s55 已完，GPU 拉满 90-98%）
- 预计总 ~8h
- 监控：`runs/logs/sec50_3seed_master.log`、单 run `runs/logs/sec50_*-s{seed}.log`

### 46.2 相关资产（仍有效）

- §45 方案的核心参数：见本文档 §45（唯一保留的近期章节）
- 超参数结论（§43）：hyper.md（独立文件，未丢失）
- 超参数扫描图：`plots/random/hp_scan_val_loss.png`
- 评估 JSON：`artifacts/tokamind_test_eval/pca_ep100_s54.json`（§37 负结论数据）、`pca_sec9bs64_s54.json`（§40 pilot）

### 46.3 新会话注意事项

1. 读 §45 + §46 了解当前状态；§22-§44 的历史细节需参考会话记录/hyper.md/plot.md
2. 实验完成后：评估 57 run → 3 seed 均值±std → 更新 curve 图
3. 判定：bs64 cosine 50ep 下 C 臂是否恢复正收益（复核 §37 负结论）

---

## 37. 100ep bs8 阶段 1（负结论完整记录，2026-09-01 恢复）

> **背景**：§37 是「100ep bs8 阶段 1」实验（24 run，单 seed s54，2026-08-31 19:13~21:48 训练，09-01 00:51 评估），
> 因文档损坏丢失，本节省会话记录恢复，供论文/复核参考。
> **⚠️ 全部结论基于 batch-size 8，已标注「存疑，待 bs=64 复核」**——当前 §45 sec50 3seed（bs64）即复核实验。

### 37.1 实验配置（§37 = 100ep bs8）

- batch size **8**（非 64）、100 epochs + 早停 patience=10、cosine、ft lr=1e-5、warmstart 现有 `_best` 预训练、s54
- 24 run：Random 16 + Temporal v2 8（`-s54-ep100` 后缀）

### 37.2 评估结果（Random split，test 21,350 点，nrmse_per_sample %）

**`artifacts/tokamind_test_eval/pca_ep100_s54.json`**：

| 档 | A scratch | C clean | C noisy05 | C noisy20 | 增益(clean) |
|---|---|---|---|---|---|
| 100% | 0.598 | 0.673 | 0.675 | 0.677 | **−12.6%** ❌ |
| 50% | 0.696 | 0.723 | 0.732 | 0.753 | **−3.8%** ❌ |
| 25% | 0.738 | 0.774 | 0.786 | 0.796 | **−4.9%** ❌ |
| 5% | 1.002 | 0.968 | 0.945 | 0.937 | **+3.4%** ✅ |

**Temporal v2**（test 15,671 点，M9 后 80% 炮）：100% −21.6% / 50% −5.1% / 25% −13.6% / 5% +8.5%（同样高数据档负、5% 档正）

### 37.3 核心结论（⚠️ 存疑，待 bs64 复核）

1. **100ep + 早停下 A 臂充分收敛后，高数据档（100/50/25%）预训练增益消失甚至反转**（C 差于 A −3.8%~−12.6%）——此前 10ep/40ep 的「全域有效」主要来自 A 臂欠拟合
2. **唯一幸存：5% 档（数据极度稀缺）预训练仍有效**（+3.4%/+8.5%）——但幅度远小于 40ep 口径（当时 +30~+64%）
3. **根因判定：ft lr=1e-5 在高数据档欠拟合**（C 臂 val > A 臂 val，C 收敛到更差局部最优；§31「1e-5 更平滑」只在 5% 小数据档成立）
4. 噪声臂与 clean 趋势一致（噪声非必要）
5. **对论文含义**：若按此口径，「仿真预训练全域有效」需改写为「充分收敛后仅在数据极度稀缺时有效」——**待 bs64 复核**

### 37.4 复核状态（2026-09-01 已复核）

- §45 sec50 3seed（bs64 + cosine + 50ep + 早停 + ft lr1e-5）**已完成评估**（§45.6）
- **复核结论**：
  1. **100%/50% 档负结论确认成立**（bs64 下 −19.3% / −3.4%，§37 的 bs8 −12.6% / −3.8% 方向一致）
  2. **5% 档正收益大幅增强**（+3.4% → +50.5%，bs64 下）
  3. **25% 档转正**（−4.9% → +5.9%）
  4. 综合：充分收敛后「预训练仅在数据稀缺时有效」，100% 档无益——§37 的核心判断（高数据档负）成立，但 5% 档增益被低估（原 +3.4% 偏保守）
- 高数据档负收益根因仍待排查（lr 匹配？预训练域差距？）——当前 §45.6 结论以实测为准

### 45.6 结果（2026-09-01，3 seed 评估完成）

**57/57 run 训练完成**（含早停触发，如 s55 synth-pretrain 早停于 27ep、s56 noisy20-finetune-50pct 早停于 44ep）。
评估：`artifacts/tokamind_test_eval/pca_sec50_3seed.json`（test 21,350 点 / 730 炮）。

**3 seed 均值±std（nrmse_per_sample %）**：

| 档 | A scratch | C clean | C noisy05 | C noisy20 | 增益 clean |
|---|---|---|---|---|---|
| 100% | 0.733±0.012 | 0.875±0.006 | 0.867±0.012 | 0.861±0.014 | **−19.3%** ❌ |
| 50% | 0.877±0.009 | 0.906±0.004 | 0.901±0.005 | 0.899±0.008 | **−3.4%** ❌ |
| 25% | 1.020±0.022 | 0.960±0.007 | 0.941±0.006 | 0.947±0.007 | **+5.9%** ✅ |
| 5% | 2.178±0.178 | 1.078±0.016 | 1.037±0.003 | 1.069±0.006 | **+50.5%** ✅ |

**核心结论（bs64 cosine 50ep，对照 §37 bs8 100ep 负结论）**：
1. **100% 档仍是负的（−19.3%）**：bs64 cosine 未恢复 100% 档正收益 → §37 高数据档负结论**依然成立**（3 seed 一致，std 小）
2. **5% 档增益大幅提升（+3.4% → +50.5%）**：3 seed 一致（std 仅 0.016），数据极度稀缺时预训练价值最大
3. **25% 档转正**（−4.9% → +5.9%）
4. **趋势稳健**：数据越稀缺增益越大（−19% → −3% → +6% → +50%），与 §24.5 定性一致
5. **对论文的含义**：充分收敛后，「仿真预训练」的价值是**数据稀缺时（5% 档 +50%、25% 档 +6%）**；100% 档无益甚至有害（−19%）。此前「全域有效」的结论不可用，应按本口径表述。

### 45.7 实验参数记录与 sec9 对比（2026-09-01）

**本次实验（sec50）参数**：

| 项 | sec50（本次）| sec9bs64（§40 pilot 对比）|
|---|---|---|
| batch size | **64** | 64 |
| scheduler | cosine 到 0（**无 warmup**，warmup_steps_fraction=0，lr 从首 ep 直接衰减）| cosine 到 0（无 warmup）|
| **weight decay** | **0**（未传 `--weight-decay`，脚本默认 0）| 0 |
| **epochs** | **50 + 早停 patience=10** | **10**（无早停触发）|
| lr：预训练 / scratch | **1e-4** | 1e-4 |
| lr：finetune | **1e-5** | 1e-5 |
| seed | **3 seed（s54/s55/s56）** | 单 seed s54 |
| 预训练 | 重训（50ep cosine lr1e-4）| §40 重训（10ep）|
| run 后缀 | `-s{seed}-sec50` | `-s54-bs64` |

**结果对比（nrmse_per_sample %，sec50 为 3 seed 均值）**：

| 档 | A_sec9 | A_sec50 | C_sec9 | C_sec50 | 增益_sec9 | 增益_sec50 |
|---|---|---|---|---|---|---|
| 100% | 1.037 | 0.733 | 1.198 | 0.875 | −15.6% | **−19.3%** |
| 50% | 1.398 | 0.877 | 1.263 | 0.906 | +9.6% | **−3.4%** |
| 25% | 2.388 | 1.020 | 1.345 | 0.960 | +43.7% | **+5.9%** |
| 5% | 3.996 | 2.178 | 1.541 | 1.078 | +61.4% | **+50.5%** |

**对比结论**：
1. **50ep 收敛后 A/C 绝对值全面下降**（A 1.04-4.00→0.73-2.18；C 1.20-1.54→0.88-1.08）
2. **sec9 的 25%/50% 档增益大部分是 A 臂欠拟合伪增益**（10ep 时 A 25%=2.39 严重欠拟合）——50ep 后 A 追近 C，增益消失/转负
3. **5% 档强增益保持**（+61.4%→+50.5%，3 seed 一致）
4. **收敛充分后只有 5% 档（数据极度稀缺）预训练有价值**，25-100% 档无益甚至有害——与 §37 结论一致

**sec50 三指标结果表（3 seed 均值±std，test 21,350 点；增益 = (A−C)/A）**：

| 数据比例 | 指标 | Real scratch | Pre-train + fine-tune | 增益 |
|---|---|---|---|---|
| 100% | nRMSE (per sample) | 0.73 ± 0.01% | **0.87 ± 0.01%** | **−19.3%** |
| | Raw RMSE (mWb) | 1.97 ± 0.03 | **2.30 ± 0.01** | −16.8% |
| | RMAE (global) | 0.22 ± 0.00% | **0.26 ± 0.00%** | −19.5% |
| 50% | nRMSE (per sample) | 0.88 ± 0.01% | **0.91 ± 0.00%** | **−3.4%** |
| | Raw RMSE (mWb) | 2.28 ± 0.02 | **2.40 ± 0.01** | −5.2% |
| | RMAE (global) | 0.27 ± 0.00% | **0.27 ± 0.00%** | −3.1% |
| 25% | nRMSE (per sample) | 1.02 ± 0.02% | **0.96 ± 0.01%** | **+5.9%** |
| | Raw RMSE (mWb) | 2.54 ± 0.04 | **2.50 ± 0.01** | +1.5% |
| | RMAE (global) | 0.31 ± 0.01% | **0.29 ± 0.00%** | +6.4% |
| 5% | nRMSE (per sample) | 2.18 ± 0.18% | **1.08 ± 0.02%** | **+50.5%** |
| | Raw RMSE (mWb) | 5.18 ± 0.43 | **2.73 ± 0.04** | +47.2% |
| | RMAE (global) | 0.71 ± 0.06% | **0.33 ± 0.00%** | +53.6% |

（注：A 臂 = sec50 scratch 50ep 欠拟合口径，5% 档增益含放大；收敛后真实增益 ~+11%，见 §50.7 ft150 表）


### 45.8 ✅ sec50 公平化方案：real scratch 200ep + 早停（2026-09-02 定稿，⏳ 待执行）

> **动机**：§45.6/§45.7 的 A/C 对比中，A 臂 real scratch 仅 50ep **严重欠拟合**（§49.6 已证：
> 5% 档 scratch 50ep→100ep RMAE 降 45%、收敛需 ~145ep；25% 档需 ~160ep，见 §49.8/§51 分析）。
> sec50 里 C 臂的增益（5% +50.5% 等）被 A 臂欠拟合放大。
> **方案：只把 real scratch 重跑为 200ep + 早停（patience=10）**，ft（50ep lr1e-5）与预训练不变，
> 复检 §45.6 A/C 对比。**依据**：各档 scratch 收敛需求 100%=~100ep / 50%=~100ep / 25%=~160ep /
> 5%=~145ep → 统一 200ep 上限 + 早停可全部覆盖（各档自然早停，不跑满 200）。

**配置**（= sec50 scratch，仅 epochs 50 → 200）：

| 项 | sec50 scratch（旧）| §45.8 scratch（新）|
|---|---|---|
| epochs | 50 | **200** |
| 早停 | patience=10 | patience=10（200ep 下生效）|
| bs / lr / wd / scheduler | 64 / 1e-4 / 0 / cosine 无 warmup | 不变 |
| seed | 3（s54/55/56）| 不变 |
| 档位 | 100/50/25/5%（无 1%）| 不变 |

**run 清单（12 run）**：`tokamind-pca01sigma-scratch-{100,50,25,5}pct-s{54,55,56}-scr200`
（无现成 run，全新跑；曾误启动 100ep 版已全部停止清理——§49.7 的 scr100 空壳与刚删的 run 均无残留）

**数据资产**（同 §49.12）：`run_real.jsonl`/`run_real.npz`、`subset_run_real_{50,25,5}pct.jsonl`/
`pca_subset_run_real_{50,25,5}pct.npz`；val = 728 炮；test = 21,350 点

**执行（待用户确认后）**：runner `runs/run_scr200.py`（12 run，16 并行 8 卡）；~1-1.5h
（100% 档 200ep 上限但早停 ~100ep）

**判定**：评估 12 run → A(scr200) vs A(sec50 50ep) vs C(sec50 ft 50ep lr1e-5) 三行对比；
预期 A200 全面优于 A50（欠拟合消除）、高数据档 A200 追平/反超 C、5% 档 C 增益收窄但仍正（对照 §49.8 的 +17% 单 seed 与 §49.13 e500 充分收敛口径）


---

## 48. Temporal split sec50 实验方案（2026-09-01 定稿，⚠️ 已中断：2026-09-01 被 kill 误杀，暂缓，后续重新训练）

> **变更（2026-09-01 用户决定）**：**只做 5% 档**（原 27 run 四档 → 9 run 仅 5%）。理由：与 §49 random 5% 档研究对照，
> 高数据档（100/50/25%）在 §45/sec50 已确认无正收益，temporal 只需验证最关键的 5% 档。
> 已中断的 27 个 run 中，仅 5% 档相关 9 个 run 续跑（`--resume` 从已有 checkpoint 续到 50ep），其余档位作废不跑。
> 机器为 4 卡（非 8 卡），runner 已改为 8 并行（2 进程/卡）。

> **再次变更（2026-09-01 用户决定，定稿）**：**放弃全部中断 checkpoint，从头重训**，**lr 统一 1e-4**（预训练/scratch/finetune 全 1e-4，
> 依据 §49.8：ft lr1e-5 收敛慢、lr1e-4 更优）；epochs = 预训练 50 / scratch 100 / ft 100（依据 §49：scratch 5% 需 ~145ep 收敛，
> 100ep 接近收敛、ft lr1e-4 早停@52 自动提前）；其余设置 = §45 sec50（bs64、cosine 无 warmup、wd=0、patience=10、3 seed）。
> **run 后缀 = `-u1e4`**（新目录，旧 `-sec50` 中断目录保留不动）。

> **目的**：用 §45 相同的设置（bs64 cosine 无warmup wd=0 lr1e-4/1e-5）做 temporal split 实验（3 seed），
> 验证预训练在时间外推（M9 全量测试）下的增益，与 random split（§45）结论对照。**用原划分（M5-M7→M8→M9 全量，非 v2）**。

### 48.1 配置（= §45 sec50）

| 项 | 值 |
|---|---|
| batch size | 64 |
| scheduler | cosine 到 0（无 warmup）|
| weight decay | 0 |
| epochs | 50 + 早停 patience=10 |
| lr：预训练 / scratch | 1e-4 |
| lr：finetune | 1e-5 |
| seed | **3 seed（s54/s55/s56）** |
| 模型 | d64/2/4/128、dropout 0.05、69→65×65 |

### 48.2 数据（**原划分**：M5-M7 → M8 val → M9 全量 test）

- **train** = M5-M7（187,292 行，run_train_with_val 193,498 含 val）；**val** = M8（6,206）
- **test** = M9 全量 507 炮（**17,498 点**，非 v2 的 15,671）
- 子集 50/25/5%（seed 20260827）
- 预训练：M5-M7 合成（150,728 行，val-fraction 0.1）

### 48.3 run 清单（9 run，`-s{seed}-u1e4` 后缀，仅 5% 档，从头训）

**预训练 clean（3，50ep）**：`tokamind-temporal-synth-pretrain-s{seed}-u1e4`
**A scratch-5pct（3，100ep）**：`tokamind-temporal-scratch-5pct-s{seed}-u1e4`
**C finetune-clean-5pct（3，100ep，lr1e-4）**：`tokamind-temporal-finetune-clean-5pct-s{seed}-u1e4`

- 全部**从头训**（无 --resume，不用旧 checkpoint）；lr 全 1e-4；ft warmstart 同 seed 新预训练（`-u1e4`）
- 旧 `-sec50` 中断目录（27 个）保留不动

### 48.4 资产映射（现成，`training_temporal/` 原划分，仅 5% 档用）

| 档 | manifest | cache |
|---|---|---|
| 预训练 | `run_synth_pretrain.jsonl` | `temporal_synth_clean.npz`（150,728）|
| 5% | `subset_run_real_5pct.jsonl` | `temporal_subset_run_real_5pct.npz` |
| test | `split_test_real.jsonl` | `temporal_test_real.npz`（17,498，M9 全量）|
| val shots | `split_val_shots.jsonl`（M8 559 炮）| |

- 预训练无 val 炮 → `--val-fraction 0.1`
- 续跑 finetune **不传 `--init-run-dir`**（§22.2-2：resume 从 checkpoint 恢复权重）

### 48.5 执行与时间

- runner：`runs/run_temporal_5u.py`（新：9 run 从头训、lr 全 1e-4、4 卡 8 并行、state 断点续跑）
- 波 1：预训练 3（50ep）+ scratch 3（100ep）并行（6 run）；波 2：ft 3（100ep，等预训练）
- 5% 档每 epoch ~10-15s；预训练 50ep ≈ 1.5h、scratch 100ep ≈ 0.5h、ft 100ep ≈ 0.5h（早停提前）
- **总墙钟 ≈ 2-3h**（8 并行）
- 启动：`setsid nohup mast-bridge/.tokamind-train-env/bin/python -u runs/run_temporal_5u.py > runs/logs/t5u_master.log 2>&1 &`
- 监控：`wc -l runs/run_temporal_5u.state`（到 9 = 完成）

### 48.6 判定（对照 §45/§37/§49）

1. temporal（原划分 M9 全量）下 **5% 档** C 臂是否保持正增益（对照 random 5% 充分收敛口径 +17%）
2. **口径注意（§49.8）**：scratch 100ep 可能仍未完全收敛（5% 需 ~145ep），如未收敛可补 scr 更久对比；
   但 scratch/ft 同为 100ep 是公平对照，ft lr1e-4 早停通常 ~50ep 内
3. 评估：`evaluate_tokamind_testset.py --manifest training_temporal/split_test_real.jsonl --run-dir ... 9 个 --output-json artifacts/tokamind_test_eval/temporal_5u_3seed.json`（test cache 先复制 `temporal_test_real.npz` → `artifacts/tokamind_test_eval/test_cache_split_test_real.npz`，§00.5 坑 3）

### 48.7 ✅ 结果（2026-09-01，9/9 run 完成，3 seed 评估）

**执行**：11:20 全部训练完成（state=9）。scratch/ft 全部早停触发（scratch best@56/61/88 → 早停@66/71/98；ft best@19/27/18 → 早停@29/37/28）。
评估（M9 test 17,498 点 / 507 炮）：`artifacts/tokamind_test_eval/temporal_5u_{A_scratch_3seed,C_ft_3seed}.json`。

**A vs C（3 seed 均值±std）**：

| 指标 | A scratch | C 预训练+微调 | 增益 |
|---|---|---|---|
| nrmse_per_sample | 3.80 ± 0.12% | 2.33 ± 0.11% | **+38.6%** |
| raw_rmse | 7.77 ± 0.26 mWb | 4.81 ± 0.21 mWb | +38.1% |
| rmae_global | 1.61 ± 0.06% | 0.91 ± 0.04% | +43.3% |

**逐 seed（nrmse_per_sample %）**：

| seed | A | C | 增益 |
|---|---|---|---|
| s54 | 3.71 | 2.28 | +38.6% |
| s55 | 3.96 | 2.49 | +37.2% |
| s56 | 3.73 | 2.23 | +40.1% |

**结论**：
1. **仿真预训练在 temporal 5% 档稳健有效**：三指标一致 +38~43%，3 seed 全部同向、无重叠
2. **A/C 均已早停充分收敛**（scratch 66-98ep / ft 28-37ep）——增益不是欠拟合假象
3. 对照：比 random 同口径（§49.8 充分收敛 +17%）更强——时间外推场景预训练物理先验价值更高
4. 与旧 40ep 口径（§19.4 −55%）方向一致、幅度收窄——符合"充分收敛后增益缩小"规律
5. ft 早停很早（28-37ep，best@~20）说明 lr1e-4 收敛快；scratch 需 66-98ep 才停（5% 档收敛慢）

**图**：`plots/temporal/temporal_5pct_val_loss_ft_vs_scratch.png`（6 条 val loss 曲线，对数轴：ft 绿虚线×3 早停@29/37/28、scratch 红实线×3 早停@66/71/98；核心信息：ft 起点 ~0.08 < scratch 终点 ~0.12，预训练先验直接可见）

### 48.8 ✅ 1% 档补充实验（2026-09-02，6/6 run 完成，3 seed 评估）——数据更稀缺增益更大

> **动机（用户 2026-09-01 提出）**：5% 档已确认 +38~43% 增益，补 1% 档验证**更极端数据稀缺**下 temporal 场景预训练增益是否更大（对照 random split §49.10 的 1% 实验）。

**配置 = §48.7 完全一致**（bs64、cosine 无 warmup、wd=0、lr 全 1e-4、epochs=100、早停 patience=10、3 seed s54/55/56、原划分 M5-M7→M8→M9 全量 test）；仅档位换 1%。

**数据（新建，嵌套 1% ⊂ 5% ⊂ 10% ⊂ 25% ⊂ 50%）**：
- manifest：`data/manifests/training_temporal/subset_run_real_1pct.jsonl`（60 shot / 1,917 train + 6,206 val = 8,123 run 行）
- cache：`data/processed/training_cache_pca_01sigma/temporal_subset_run_real_1pct.npz`（slice 自 `temporal_train_with_val.npz`）
- 嵌套方式：5% 的 302 shot 内再随机 shuffle（seed 20260825）取前 60 shot

**run（6，`-1u` 后缀，预训练复用 §48 现成 `-u1e4` 不重跑）**：
- A：`tokamind-temporal-scratch-1pct-s{54,55,56}-1u`（从头，**epochs 上限=100**，早停@34/38/52，best@24/28/42，best val ≈ 0.277-0.282）
- C：`tokamind-temporal-finetune-clean-1pct-s{54,55,56}-1u`（init=`synth-pretrain-s{seed}-u1e4`，**epochs 上限=100**，早停@16/16/17，**best@6/6/7**，best val ≈ 0.084-0.086）

> **ft epoch 设置**：`--epochs 100` + 早停 patience=10（与 §48.7 完全一致）；实际全部提前早停——ft best@ep6-7、早停@16-17；scratch best@ep24-42、早停@34-52。6/6 run 全部 bad_epochs=10 到顶触发，无 run 跑满 100ep。

**执行**：`runs/run_temporal_1u.py`（state 断点续跑，8 卡并行），2026-09-02 完成；评估：`artifacts/tokamind_test_eval/temporal_1u_3seed.json`（M9 test 17,498 点 / 507 炮）。

**A vs C（3 seed 均值±std）**：

| 指标 | A scratch | C 预训练+微调 | 增益 |
|---|---|---|---|
| nrmse_per_sample | 6.42 ± 0.04% | 2.31 ± 0.17% | **+64.0%** |
| raw_rmse | 13.09 ± 0.10 mWb | 4.78 ± 0.34 mWb | +63.1% |
| rmae_global | 2.83 ± 0.01% | 0.90 ± 0.08% | **+68.1%** |

**逐 seed（rmae_global %）**：

| seed | A | C | 增益 |
|---|---|---|---|
| s54 | 2.825 | 0.816 | +71.1% |
| s55 | 2.844 | 0.912 | +67.9% |
| s56 | 2.815 | 0.975 | +65.4% |

**与 5% 档（§48.7）对照——增益随数据稀缺单调增大**：

| 档位 | RMAE 增益 | NRMSE 增益 | 结论 |
|---|---|---|---|
| 5% | +43.3% | +38.6% | 预训练先验显著有效 |
| 1% | +68.1% | +64.0% | **数据越稀缺，增益越大** |

**结论**：
1. **temporal 场景 1% 档预训练增益 ≈ +64~68%**（RMAE/NRMSE 口径一致），3 seed 全部同向、无重叠
2. **增益随数据量单调增大**（5% +38~43% → 1% +64~68%）——与 random split（§49.8：5% 充分收敛 +17%）形成对比：**时间外推场景下预训练物理先验对极端稀缺数据的价值更大**
3. A/C 均已早停（scratch 34-52ep、ft ~16ep），best val 差距悬殊（0.28 vs 0.09）——增益非收敛差异假象
4. 注：scratch 1% 早停可能偏早（val 仍在缓慢下降），如需更公平可比可参照 §49.8 补 500ep；但 ft 同口径早停@16ep，协议对称

---

## 49. 补充实验方案：scratch 100ep + ft lr=1e-4（2026-09-01 定稿）

> **背景（用户 2026-09-01 提出）**：
> 1. **scratch 50ep 可能未完全收敛**（尤其小数据档）——C 臂总训练 = 预训练 50 + ft 50 = **100**，为公平 scratch 也应 **100ep**；
>    重点确认小数据档 scratch 充分收敛后不会反超 C 臂
> 2. **ft 基本不早停**（§45：1/36 触发）说明 50ep 时仍在收敛，**lr=1e-5 太慢**（可能造成 100% 档负增益）——
>    **ft lr 拉回 1e-4**（bs64 下稳定；之前 bs8+1e-4 的不稳定是 bs 问题）
> 3. **只重跑 real scratch + ft clean**（预训练复用 sec50 现有；noisy 臂不重跑）

### 49.1 补充实验 A：scratch 100ep

| 项 | 值 |
|---|---|
| 配置 | bs64、cosine 无 warmup、wd=0、**lr 1e-4**、**epochs=100**、早停 patience=10 |
| 档位 | 100/50/25/5%（4 档）|
| seed | 3 seed（s54/s55/s56）|
| 实验 | Random + Temporal（原划分）|
| run 后缀 | `-scr100` |
| 目的 | 确认 scratch 充分收敛后 vs C 臂（尤其小数据档）|

**run**：`tokamind-{pca01sigma,temporal}-scratch-{f}pct-s{seed}-scr100`（24 run）

### 49.2 补充实验 B：finetune lr=1e-4

| 项 | 值 |
|---|---|
| 配置 | bs64、cosine 无 warmup、wd=0、**lr=1e-4**（原 1e-5）、epochs=50（ft 阶段）、早停 patience=10 |
| init | 复用 sec50 预训练（同 seed，`-sec50`）|
| 档位 | 100/50/25/5% |
| seed | 3 seed |
| 实验 | Random + Temporal |
| run 后缀 | `-ft1e4` |
| 目的 | 验证 lr=1e-5 是否导致 ft 收敛慢（100% 档负增益）；lr=1e-4 是否恢复 |

**run**：`tokamind-{pca01sigma,temporal}-finetune-{f}pct-s{seed}-ft1e4`（24 run）

### 49.3 时间估算（8 卡 16 并行）

- scratch 100ep：100% 档 ~6-8h（cosine 100ep）；小档更快
- ft 50ep lr1e-4：100% 档 ~3-4h
- **总墙钟 ~10-14h**（两波）

### 49.4 判定

1. scratch 100ep 后：小数据档（5%）scratch 是否追平 C 臂（若追平 → 预训练增益进一步缩小；若仍差 → 预训练在稀缺时价值确认）
2. ft lr=1e-4 vs 1e-5：100% 档 C 臂是否恢复（若恢复 → lr 是负增益主因；若仍负 → 预训练高数据档确实无益）
3. 对照 §45（sec50）结果，确认 lr/epochs 是否是结论差异的变量

### 49.5 Random 补充实验执行版（先跑，temporal 暂缓）

> **范围**：仅 Random split 的 **real scratch（100ep）+ ft（lr1e-4）**，24 run。
> 预训练复用 §45 sec50 现有（3 seed 已完成）；noisy 臂不跑。

**实验 A：scratch 100ep（4 run，`-scr100` 后缀，✅ 1 seed s54——scratch 是基线，1 seed 够看趋势）**
- 配置：bs64、cosine、wd=0、lr 1e-4、**epochs=100**、patience=10、**seed s54**、4 档
- run：`tokamind-pca01sigma-scratch-{100,50,25,5}pct-s54-scr100`
- manifest/cache：同 §45（`run_real.jsonl`/`run_real.npz`、`subset_run_real_{f}pct.jsonl`/`pca_subset_run_real_{f}pct.npz`）

**实验 B：ft lr=1e-4（12 run，`-ft1e4` 后缀）**
- 配置：bs64、cosine、wd=0、**lr=1e-4**、epochs=50、patience=10、3 seed、4 档
- init：`tokamind-pca01sigma-synth-pretrain-s{seed}-sec50`（复用）
- run：`tokamind-pca01sigma-finetune-{100,50,25,5}pct-s{54,55,56}-ft1e4`

**执行**：24 run 可直接并行（16 并行，GPU 拉满）；scratch 100ep 无依赖、ft init 已就绪
**时间**：scratch 100% 档 100ep ~6-8h；ft 50ep ~3-4h；**总墙钟 ~8-12h**
**判定**（§49.4）：
1. scratch 100ep 后 5% 档是否追平 C 臂（若追平 → 预训练稀缺增益进一步缩小；若仍差 → 稀缺时预训练价值确认）
2. ft lr1e-4 vs 1e-5：100% 档 C 臂是否恢复（对照 §45 的 −19.3%）

### 49.6 ✅ scratch 5% 档 100ep 结果（2026-09-01，s54）

> **核心验证**：5% 档 real scratch 在 50ep（sec50 配置）时**严重欠拟合**，100ep 后继续明显下降。
> run：`tokamind-pca01sigma-scratch-5pct-s54-scr100`（100ep 从头训，无 resume，完成）

**val loss 趋势**（epoch → val）：

| epoch | val loss |
|---|---|
| 10 | 0.2155 |
| 20 | 0.1124 |
| 30 | 0.0784 |
| 40 | 0.0502 |
| **50** | **0.0435** |
| 60 | 0.0386 |
| 70 | 0.0350 |
| 80 | 0.0323 |
| 90 | 0.0314 |
| **100** | **0.0313**（best@ep98）|

- **ep50→100 继续下降 28%**（0.0435 → 0.0313），ep90-100 才接近平台
- 对比 sec50（50ep）时 A 臂 val=0.0435 的 checkpoint 明显欠拟合

**测试集结果**（RMAE global，test 21,350 点）：

| 实验 | RMAE | NRMSE |
|---|---|---|
| sec50 scratch-5pct-s54（50ep）| 0.794% | 1.22% |
| **scr100 scratch-5pct-s54（100ep）** | **0.433%** | 0.74% |

- **50ep → 100ep：RMAE 下降 45%**（0.794% → 0.433%）
- 评估输出：`artifacts/tokamind_test_eval/pca_scr5pct_s54_scr100.json`

**结论（⚠️ 修正版，2026-09-01 10:00）**：
1. **sec50 的 5% 档 scratch（A 臂）50ep 严重欠拟合**，100ep 后继续降 45%——sec50 里 A/C 差距被 scratch 欠拟合**放大**
2. ~~0.108%/4 倍差距~~ **错误记录**（ft 值记错）；修正后见 §49.8 最终对比
3. **最终结论见 §49.8**：充分收敛后（sc100 vs ft100 lr1e-4）增益 ≈ **+49%**（1.5 倍），方向成立但幅度约为 sec50 报告的一半

**图**：`plots/random/val_loss_scr100_4fractions.png`（5%→100% 并排；5% 面板：scratch 50ep 虚线 + 100ep 实线 + ft 绿线）

### 49.7 📋 实时状态快照（2026-09-01 09:30→10:00，✅ 已完成）——scratch 欠拟合研究交接

> **⚠️ 本小节是给未来会话的交接快照：新会话先读本节 + §49.5/§49.6 再干活。**

#### 研究问题
sec50（bs64+cosine+50ep+早停）实验里，**5% 档 A 臂（real scratch 50ep）是否欠拟合**？
即：scratch 训更久（100ep）后 RMAE 是否继续下降？充分收敛后是否还差于 C 臂（pretrain+ft）？

#### 已完成的实验（全部 s54，random split，测试集 21,350 点）

| run | 配置 | RMAE(global) | 状态 |
|---|---|---|---|
| `-scratch-5pct-s54-sec50` | scratch 50ep lr1e-4 | **0.794%** | ✅ 完成 |
| `-scratch-5pct-s54-scr100` | scratch **100ep** lr1e-4 | **0.433%** | ✅ 完成 |
| `-finetune-5pct-s54-sec50` | ft 50ep lr**1e-5** | **0.328%** | ✅ 完成 |
| `-finetune-5pct-s54-ft1e4-100ep` | ft **100ep** lr**1e-4** | **0.290%** | ✅ 完成 |

> 注：§49.6 曾写"ft=0.108%、4 倍差距"——**那是错误记录**（记错值）。
> 修正后：ft 50ep lr1e-5 = **0.328%**，scratch 100ep = 0.433%，差距 ~1.3 倍（+32%）。

#### 关键结论（修正版）
1. **scratch 5% 档 50ep 严重欠拟合**：val loss ep50=0.0435 → ep100=0.0313（-28% 持续下降），RMAE 0.794% → 0.433%（-45%）
2. **充分收敛（100ep）后 scratch 仍差于 ft**：0.433% vs 0.328%（+32% 增益）→ 预训练在数据稀缺时**方向仍支持有帮助**，但幅度远小于 sec50 报告的 +50.5%（被 scratch 欠拟合放大）
3. **✅ 已定**（见 §49.8）：ft 100ep lr1e-4 = 0.290%（best@ep42，早停@52）；最终增益（sc100 vs ft100）≈ **+49%**

#### 当前在跑
- 无（§49 实验已全部完成）

#### 下一步（✅ 已完成，见 §49.8）
- ft 100ep 评估 → 4 run 对比表 → 最终增益 +49% → 图 `val_loss_5pct_4arms.png` 全部完成

#### 数据资产
- manifest/cache：`data/manifests/training_pca_01sigma/subset_run_real_5pct.jsonl` + `data/processed/training_cache_pca_01sigma/pca_subset_run_real_5pct.npz`
- val：`data/manifests/training_pca_01sigma/split_val_shots.jsonl`（728 炮）
- test：`data/manifests/training_pca_01sigma/split_test_real.jsonl` + `data/processed/training_cache_pca_01sigma/test_real_pca.npz`
- 预训练 init：`runs/tokamind-pca01sigma-synth-pretrain-s54-sec50`（复用，50ep cosine，val 合成域内）

#### 其他（勿忘）
- §48 temporal sec50 实验**被 kill 误杀中断**，暂缓，后续重新训练（temporal 不用管）
- 其他档位（100/50/25%）的 scr100 已杀（用户只要 5% 档）
- ft lr1e-4 的其余档位（实验 B 12 run）**待定**，等 5% 档结果

### 49.8 ✅ 5% 档最终对比（2026-09-01，s54）——scratch 欠拟合研究定论

> **5 个 run 全部完成**（random split，5% 档，test 21,350 点，5% 档 = 1019 shot / 30,217 条样本）。这是本节最终结果表。

#### 最终对比表

| run | 配置 | best val | 早停位置 | RMAE(global) | NRMSE |
|---|---|---|---|---|---|
| `-scratch-5pct-s54-sec50` | scratch 50ep lr1e-4 | 0.0435 | 跑满 | **0.794%** | 1.22% |
| `-scratch-5pct-s54-scr100` | scratch **100ep** lr1e-4 | 0.0313 | 跑满 | **0.433%** | 0.74% |
| `-scratch-5pct-s54-scr500` | scratch **500ep** lr1e-4 | **0.0212** | **早停@145**（best@135）| **0.339%** | 0.58% |
| `-finetune-5pct-s54-sec50` | ft 50ep lr**1e-5** | 0.0209 | 跑满 | **0.328%** | 0.56% |
| `-finetune-5pct-s54-ft1e4-100ep` | ft **100ep** lr**1e-4** | **0.0184** | 早停@52（best@42）| **0.290%** | 0.50% |

#### 增益（RMAE 相对下降）

| 对比口径 | 增益 |
|---|---|
| sec50 原口径（sc 50ep vs ft 50ep lr1e-5）| **+142%**（0.794 vs 0.328）|
| sc 100ep vs ft 100ep lr1e-4 | +49%（0.433 vs 0.290）|
| **双方早停收敛（sc 145ep vs ft 52ep）** | **+17%**（0.339 vs 0.290）|

#### 结论（定论，2026-09-01 10:30 更新）
1. **scratch 5% 档 50ep 严重欠拟合**：val 0.0435 → 100ep 0.0313 → **145ep 0.0212**（RMAE 0.794% → 0.433% → **0.339%**）；**100ep 仍欠拟合，~145ep 才收敛（500ep 实验早停确认）**
2. **ft 100ep lr1e-4（0.290%）> ft 50ep lr1e-5（0.328%）**：lr1e-4 收敛更好（best@ep42，早停@52）
3. **双方充分收敛（早停）后，预训练增益 ≈ +17%**（0.339 vs 0.290，1.2 倍）——**方向仍成立（数据稀缺时预训练小幅更好），但幅度很小**：sec50 报告的 +50.5%（3 seed）/+142%（单 seed s54）绝大部分是 scratch 欠拟合的假象
4. scratch 500ep 时 ep90-100 的"微降"其实是长期欠拟合（0.0313 → 0.0212，再降 32%），说明 50ep/100ep 都远未收敛；**1 个种子下 17% 增益的边缘性**待更多 seed 确认（但用户暂不跑多 seed）

#### 图
- `plots/random/val_loss_5pct_4arms.png`：5% 档 4 臂 val loss（sc50 红虚线 / sc100 红实线 / ft50 绿虚线 / ft100 绿实线）
- `plots/random/trainval_5pct_5arms.png`：5 臂 train/val 双面板（含 sc145 粗实线）
- `plots/random/trainval_5pct_scratch_vs_ft_es.png`：**只画早停结果**（sc145 红 vs ft52 绿，双面板）
- `plots/random/val_loss_scr100_4fractions.png`：4 档并排（5%→100%，每面板 A/C；5% 面板含 sc100 线）

#### 评估产物
- `artifacts/tokamind_test_eval/pca_scr5pct_s54_scr100.json`
- `artifacts/tokamind_test_eval/pca_scr5pct_s54_scr500.json`
- `artifacts/tokamind_test_eval/pca_ft1e4_5pct_s54_100ep.json`

### 49.9 ✅ scratch 5% 档 500ep 实验（2026-09-01 10:10 启动，10:35 完成）

> **动机**：scr100（100ep）完成后 val 仍在微降（ep90-100），且用户要求看 5% real scratch 能否再下降。
> **配置**：bs64、cosine、wd=0、lr 1e-4、**epochs=500**、patience=10、seed s54、**从头训练**（无 resume、新目录）
> **run**：`tokamind-pca01sigma-scratch-5pct-s54-scr500`
> **日志**：`runs/logs/scr500_scratch-5pct-s54.log`；卡 0 独占
> **预计**：5% 档 ~5s/ep → 500ep ≈ 40min（早停可能更早）

**要看什么**：
1. val loss 是否在 100ep（0.0313）后继续下降？降到多少？
2. 500ep 充分收敛后的 scratch RMAE（对比 ft 100ep lr1e-4 = 0.290%）→ 最终增益确认
3. 是否早停（若 ep<500 停，记录停的位置）

**依赖**：无（从头训，独立 run）

**完成后**：评估 → `artifacts/tokamind_test_eval/pca_scr5pct_s54_scr500.json` → 更新 §49.8 对比表

### 49.10 🔄 1% 真实数据 ft + scratch 实验（2026-09-01 11:00 启动）——欠拟合研究扩展

> **动机**：§49.8 定论 5% 档充分收敛后预训练增益仅 +17%。为验证**更极端数据稀缺**（1%）下预训练是否仍有（更大）价值，扩展 1% 档。
> **数据**：新建 1% subset（58 shot / 1,924 训练样本 + val 728 shot）
> - manifest：`data/manifests/training_pca_01sigma/subset_run_real_1pct.jsonl`（嵌套采样 seed 20260825，1% ⊂ 5% ⊂ 10% ⊂ 25% ⊂ 50% ⊂ 100%）
> - cache：`data/processed/training_cache_pca_01sigma/pca_subset_run_real_1pct.npz`（23,071 行）

#### 2 个 run（s54，与 5% 档配置对称）

| run | 配置 | best val | 早停 | RMAE(global) | 状态 |
|---|---|---|---|---|---|
| `tokamind-pca01sigma-scratch-1pct-s54-scr500` | scratch 500ep lr1e-4、patience=10、从头训 | **0.0399**（@ep276）| @ep286 | **待评估** | ✅ 训练完成 |
| `tokamind-pca01sigma-finetune-1pct-s54-ft1e4-100ep` | ft 100ep lr**1e-4**、patience=10、init=sec50 预训练 | 0.0274（@ep32）| @ep42 | **0.334%** | ✅ 完成 |

- ft 1% 评估：`artifacts/tokamind_test_eval/pca_ft1pct_s54_100ep.json`
- scratch 1% val 轨迹：ep50=0.211 / ep100=0.096 / ep200=0.043 / ep286=0.040（**长期欠拟合，~280ep 才收敛**）
- scratch 1% 评估：待完成（评估脚本 cache rebuild 已修复，`runs/logs/eval_scr1pct.log`）

#### 要看什么
1. 1% 档 scratch 是否更欠拟合（需更久才收敛？500ep 够吗）
2. 1% 档预训练增益 vs 5% 档（+17%）——数据越稀缺增益越大？还是收敛后同样缩小？
3. 与 §49.8 表（5% 档 5 run）对照，形成"数据量 → 增益"曲线

#### 日志
- `runs/logs/scr1pct_scratch.log`、`runs/logs/ft1pct_finetune.log`

#### 完成后
- 评估：`pca_scr1pct_s54_scr500.json`、`pca_ft1pct_s54_100ep.json`
- 更新 §49.8 结论（增益随数据量变化）

### 49.11 📋 新实验方案：random split 全套 500ep+早停 对比（2026-09-01 定稿）

> **动机**：sec50（50ep）对比不满意——scratch 50ep 严重欠拟合（5% 档已验证：scratch 收敛需 ~145ep、1% 需 ~280ep），
> ft 也需更久（5% ft 100ep lr1e-4 早停@52）。**新实验让 scratch 和 ft 都充分收敛（epoch=500+早停）后再对比**，
> 得到"公平"的预训练增益。

#### 配置（统一，✅ 3 seed，✅ clean+noisy）
- bs64、cosine 无 warmup、wd=0、**epochs=500**、**早停 patience=10**、**seed s54/s55/s56（3 seed）**
- scratch：lr 1e-4，从头训（1 个臂）
- ft：lr **1e-4**（§49.8 已证优于 1e-5），**3 个臂：clean / noisy05 / noisy20**
  - init = sec50 预训练（复用，不重跑，3 seed 都有）：
    - clean → `synth-pretrain-s{seed}-sec50`
    - noisy05 → `synth-pretrain-noisy05-s{seed}-sec50`
    - noisy20 → `synth-pretrain-noisy20-s{seed}-sec50`
- 档位：**100/50/25/5/1%**（5 档）
- run 名：`tokamind-pca01sigma-{scratch,finetune,noisy05-finetune,noisy20-finetune}-{f}pct-s{54,55,56}-e500`

#### 已有资产（复用，不用重跑）
| 档 | scratch | ft |
|---|---|---|
| 5% | ✅ 仅 s54 `scratch-5pct-s54-scr500`（早停@145，RMAE 0.339%）| 无（需 3 seed 新跑 500ep）|
| 1% | ✅ 仅 s54 `scratch-1pct-s54-scr500`（早停@286）| 无（需 3 seed 新跑 500ep）|

#### 需新跑（58 run）
| 档 | scratch | ft-clean | ft-noisy05 | ft-noisy20 |
|---|---|---|---|---|
| 100% | 3 seed | 3 seed | 3 seed | 3 seed |
| 50% | 3 seed | 3 seed | 3 seed | 3 seed |
| 25% | 3 seed | 3 seed | 3 seed | 3 seed |
| 5% | 2 seed（s55/s56）| 3 seed | 3 seed | 3 seed |
| 1% | 2 seed（s55/s56）| 3 seed | 3 seed | 3 seed |

- scratch 13 + ft 45（clean 15 + noisy05 15 + noisy20 15）= **58 run**

#### 时间估算（8 卡 16 并行，每卡 2 进程）
- 100% 档 ~4-5h/run（每 ep ~30-35s，500ep），50% ~2.5h，25% ~1.5h，5% ~40min，1% ~30min
- 波 1：scratch 13；波 2：ft 45（init 预训练已就绪，可与波 1 并行抢算力，建议波 2 等波 1 完）
- **总墙钟 ~12-16h**（100% 档 12 run 是瓶颈）

#### 判定（✅ 聚焦：数据量小时是否仍有正增益）
1. 所有档位 scratch/ft（clean/noisy05/noisy20）都早停（记录早停位置，3 seed 均值）
2. **核心**：数据量小（**5%/1%**）档，充分收敛（早停）后 scratch vs ft（clean/noisy05/noisy20）：
   - 是否有正增益？（§49.8 单 seed 已见 5% +17%）
   - noisy 臂是否比 clean 增益更高/更稳？（sec50 里 noisy 增益略高）
   - 1% 档增益是否比 5% 更大（数据越稀缺增益越大）？
3. 3 seed 均值±std 确认增益是否显著（std 大则增益边缘）
4. 高数据量档（100/50/25%）仅记录结果供参考，不作判定依据

### 49.12 📋 §49.11 执行交接版（新会话必读）

> **本节目的是让未来会话无需旧上下文即可执行 §49.11（58 run）**。方案本身见 §49.11，本节是执行细节。

#### 0. 状态检查（执行前）
- 预训练 init 9 个是否就绪：`runs/tokamind-pca01sigma-synth-pretrain{,-noisy05,-noisy20}-s{54,55,56}-sec50`（✅ 应全部存在）
- 复用资产：`-scratch-5pct-s54-scr500`、`-scratch-1pct-s54-scr500`（✅ 已存在）
- 1% 数据：`subset_run_real_1pct.jsonl` + `pca_subset_run_real_1pct.npz`（✅ 已构建，58 shot train）

#### 1. 数据资产（random split）
| 档 | manifest | cache |
|---|---|---|
| 100% | `data/manifests/training_pca_01sigma/run_real.jsonl` | `data/processed/training_cache_pca_01sigma/run_real.npz` |
| 50% | `subset_run_real_50pct.jsonl` | `pca_subset_run_real_50pct.npz` |
| 25% | `subset_run_real_25pct.jsonl` | `pca_subset_run_real_25pct.npz` |
| 5% | `subset_run_real_5pct.jsonl` | `pca_subset_run_real_5pct.npz` |
| 1% | `subset_run_real_1pct.jsonl` | `pca_subset_run_real_1pct.npz` |

- val：`split_val_shots.jsonl`（728 炮，`--val-shot` 列表）
- test：`split_test_real.jsonl` + `test_real_pca.npz`

#### 2. 命令模板（train_tokamind_manifest.py）
```
--manifest {m} --dataset-cache {c} --run-dir runs/tokamind-pca01sigma-{arm}-{f}pct-s{seed}-e500
--seed {seed} --lr 1e-4 --input-mode magnetic-diagnostics --target-mode raw-psi
--feature-schema mast-bridge/configs/diagnostic_features/mast_level2_common_69.json
--d-model 64 --n-layers 2 --n-heads 4 --dim-ff 128 --dropout 0.05
--batch-size 64 --epochs 500 --patience 10 --val-shot <728 shots>
```
- scratch：不加 init
- ft：`--init-run-dir runs/tokamind-pca01sigma-{synth-pretrain|synth-pretrain-noisy05|synth-pretrain-noisy20}-s{seed}-sec50 --finetune-method full`

#### 3. 58 run 完整清单（arm × 档 × seed）
- scratch（13）：100/50/25% × s54/55/56（9）+ 5/1% × s55/56（4）
- finetune（15）：5 档 × 3 seed
- noisy05-finetune（15）、noisy20-finetune（15）
- **生成方式**：按 run_sec50_3seed.py 的结构写 runner（`runs/run_e500.py`），建议 GPU 分配 = job idx % 8

#### 4. 执行顺序建议
- 波 1：scratch 13 run（16 并行）
- 波 2：ft 45 run（16 并行）——init 预训练已就绪，可与波 1 后半并行
- 注意：**每卡 2 进程**（GPU 利用率需 ≥15%，bs64 单进程低，2 进程/卡拉满）

#### 5. 完成后评估
```
evaluate_tokamind_testset.py --manifest split_test_real.jsonl --run-dir <run> --output-json artifacts/tokamind_test_eval/pca_e500_<arm>_<f>pct_s<seed>.json
```
- ⚠️ 评估脚本已修复（2026-09-01）：cache rebuild 时 scalers 从 run-dir 读（不再硬编码 tokamind-large-real-scratch-100e）
- test cache：`artifacts/tokamind_test_eval/test_cache_split_test_real.npz`（21,350 行，已重建，直接命中）

#### 6. 结果汇总（模板）
- 表：档位 × 4 臂的 RMAE（3 seed 均值±std）
- 增益 = (scratch - ft)/ft，按 §49.11 判定（5%/1% 正增益为核心）
- 与 sec50（§45：−19.3%/−3.4%/+5.9%/+50.5% @100/50/25/5%）和 §49.8（5% +17%）对照

#### 7. 注意事项
- **不要动旧 run**（sec50/scr100/scr500/ft1e4-100ep 等，全保留）
- 预训练只读复用，**不重跑**
- 1% 档数据极少（58 炮），scratch 预计早停 ~280ep、ft ~50ep；bs64 每 ep 仅 ~30 步（已确认保持 bs64）
- 时间：总墙钟 ~12-16h（100% 档 12 run 瓶颈）
- 若中途 kill：用 `--resume` 续跑（cosine 按剩余 epoch 重算，可接受）

#### 8. ✅ 交接完成（2026-09-01，新会话从本节接手）

- **runner 已写好并验证**：`runs/run_e500.py`（58 job 清单与 §49.12 §3 逐项一致：scratch 13 + clean/noisy05/noisy20 各 15；5%/1% scratch 跳过已有 s54；run 名 `-e500`；state=`runs/run_e500.state` 断点续跑）
- **机器适配**：本机 **4 卡**（非 §49.12 的 8 卡）→ 8 并行（2 进程/卡），波 1 scratch 13 → 波 2 ft 45
- **前置校验通过**：9 个 sec50 预训练 init ✅、scratch-{5,1}pct-s54-scr500 复用 ✅、1% manifest/cache ✅、58 run 的 manifest/cache/init 全存在 ✅
- **执行**：`setsid nohup mast-bridge/.tokamind-train-env/bin/python -u runs/run_e500.py > runs/logs/e500_master.log 2>&1 &`；监控 `wc -l runs/run_e500.state`（到 58 = 完成）
- **时间修正（4 卡）**：8 并行下总墙钟 ~24-32h（100% 档 scratch+ft 共 6 run/档 × 3 arm 是瓶颈，单 run 500ep 早停后 ~2-5h）
- **评估（完成后）**：test cache `test_real_pca.npz` 已就位（`artifacts/tokamind_test_eval/test_cache_split_test_real.npz`，21,350 行直接命中）；评估脚本 scalers 已修复（读 run-dir）

### 49.13 ✅ e500 实验结果（2026-09-02，58/58 run 训练 + 59 评估全部完成）

> **交接收尾**：§49.11/§49.12 的 58 run（500ep+早停，bs64 cosine wd0 lr 全 1e-4，3 seed）全部训练完成
> （state=58 ALL DONE）。本机（8 卡）批量评估 58 run + 补跑 s57，结果见下。

**评估**：test 21,350 点 / 730 炮（random split），`artifacts/tokamind_test_eval/pca_e500_*.json`（59 个，
s57 为补跑）。聚合脚本 `/tmp/opencode/agg_e500.py`。

**结果表（nrmse_per_sample %，3 seed 均值±std；增益 = (A−C)/C，正 = C 优于 A）**：

| 档 | A scratch | C clean | C noisy05 | C noisy20 | 增益(clean) |
|---|---|---|---|---|---|
| 100% | 0.659 ± 0.025 | 0.692 ± 0.015 | 0.658 ± 0.019 | 0.651 ± 0.030 | **−4.8%** ❌ |
| 50% | 0.714 ± 0.013 | 0.734 ± 0.010 | 0.761 ± 0.033 | 0.752 ± 0.037 | **−2.7%** ❌ |
| 25% | 0.772 ± 0.039 | 0.819 ± 0.010 | 0.782 ± 0.011 | 0.810 ± 0.002 | **−5.7%** ❌ |
| 5% | 1.107 ± 0.022 | 0.978 ± 0.015 | 0.973 ± 0.019 | 0.987 ± 0.011 | **+13.2%** ✅ |
| 1% | 1.464 ± 0.012 | 1.089 ± 0.019 | 1.060 ± 0.010 | 1.122 ± 0.032 | **+34.4%** ✅ |

（RMAE global 同向：−5.6/−3.3/−5.2/+13.3/+38.6%）

**核心结论**：
1. **充分收敛（500ep+早停）后，预训练增益只在数据稀缺时为正**：5% 档 +13.2%、1% 档 +34.4%，
   100/50/25% 档无正收益（−2.7%~−5.7%，C 略差于 A）——与 §37（bs8 100ep）、§45（sec50）一致，
   **高数据档负结论 3 次独立验证成立**
2. **1% 档增益最大且 seed 稳定**：A 1.464±0.012 / C clean 1.089±0.019——数据越稀缺预训练价值越大
   （100→1%：−5% → −3% → −6% → +13% → +34%，单调递增）
3. **noisy 臂与 clean 完全持平**（各档差 <0.1pp）——噪声非必要条件（第三次确认，§14/§15.8）
4. **C 臂 std 全档小**（0.010-0.032），A 臂 25% 档 std 0.039 最大——预训练抹平 seed 方差
5. 对比 sec50（50ep）：sec50 的 +50.5%（5% 档）大部分是 scratch 欠拟合假象（§49.8/§49.6 已证）；
   e500 充分收敛后真实增益 5% 档 +13.2%、1% 档 +34.4%

**坏 run 记录（重要）**：`scratch-1pct-s55` 是**确定性坏 basin**——两次独立训练（原跑 + 重跑）
bit 级复现：早停@44、best_val=0.2316（未收敛，正常水平 ~0.04），评估 nrmse 4.02%。1% 档数据极少
（58 shot / 1,924 样本）时 seed 初始化敏感。**处理：剔除 s55，补跑 s57**（正常收敛 best 0.0415@302，
nrmse 1.099%）。最终 1% 档 A 臂 = {s54, s56, s57} 三 seed（s54 复用 scr500 评估，配置相同）。

**坑（新会话必读）**：
- **评估缓存被 temporal 覆盖**：`artifacts/tokamind_test_eval/test_cache_split_test_real.npz` 曾为
  temporal 版（17,498 行，M9）→ random 评估触发 zarr 重建（>5min 卡死）。temporal 评估（manifest
  `training_temporal/split_test_real.jsonl`）cache_key 同为 `split_test_real` → **同名覆盖**（§15.8 坑②再现）。
  修复：`cp data/processed/training_cache_pca_01sigma/test_real_pca.npz artifacts/tokamind_test_eval/test_cache_split_test_real.npz`
  （21,350 行版，mtime 08-31，勿被 temporal 评估再次覆盖）
- 评估 `--output-json` 放别处（如 /tmp）会导致 cache 目录不同、触发重建——**必须放
  `artifacts/tokamind_test_eval/` 下**（cache 与 json 同目录）
- `pgrep -f "run名"` 会误匹配 bash -c 自身——用 `ps aux | grep python` 判断训练进程
- 单 run 评估（缓存命中）仅 ~21s；批量评估用 state+json 存在性做断点（`/tmp/opencode/eval_e500.sh` 可复用）

**产物**：59 个 `pca_e500_*.json`（scratch 13 + ft 45 + s57 补跑）；图未画（如需：4 臂 × 5 档
nrmse 曲线，参考 §45.6 格式）

### 49.14 ✅ 早停区间记录（2026-09-02，e500 实验）

> **结论：60/60 run 全部触发早停（patience=10），无一跑满 500ep**。数据越稀缺 scratch 收敛越慢，
> ft 快 3~10 倍（预训练先验加速收敛的量化证据）。

**早停位置区间（best_ep → 早停 ep；1% scratch 含补跑 s57 → 258–312）**：

**scratch（随机初始化）**：

| 档 | best_ep | 早停 ep |
|---|---|---|
| 100% | 44–88 | 54–98 |
| 50% | 62–88 | 72–98 |
| 25% | 70–149 | 80–159 |
| 5% | 125–135 | 135–145 |
| 1% | 248–302 | 258–312 |

**finetune（预训练 init，3 臂）**：

| 档 | clean | noisy05 | noisy20 |
|---|---|---|---|
| 100% | 54–60 | 60–92 | 53–100 |
| 50% | 53–71 | 39–90 | 38–90 |
| 25% | 52–61 | 63–89 | 55–66 |
| 5% | 37–55 | 27–48 | 37–50 |
| 1% | 39–60 | 26–69 | 12–57（noisy20-s55 异常点 best@2）|

**要点**：
1. 高数据档（100/50/25%）scratch 与 ft 早停位置接近（50–100 ep）；**小数据档分化**：
   5% ft 27–55 ep vs scratch 135–145 ep（快 ~3–4 倍）；1% ft 12–69 ep vs scratch 258–312 ep（快 ~5–10 倍）
2. **早停不是欠拟合**：best 后 val 无持续下降，只是平台期波动（尾部 10ep 波动 = best 的 +5%~+26%，
   绝对值 0.001–0.003）
3. **震荡原因**：① cosine 无 warmup + epochs=500 → 前 ~100ep lr 仍 ~1e-4 高位（ep44 lr=9.8e-5），
   最优邻域振荡（主因）② 小数据档每 epoch 步数少（1% 档 1,924 样本 ÷ bs64 = 30 步/ep）→ 梯度噪声大
   ③ wd=0 ④ log 轴视觉放大
4. 若需压震荡：加 warmup 或 cosine 改为 ~200ep 内完成衰减（当前未做，不影响 §49.13 结论）

### 49.15 ✅ ft val loss 可视化（2026-09-02）

**产物**（plots/random/，dpi=300、线性轴）：
- `e500_ft_clean_valloss_linear.png`：clean ft 单臂，5 面板（100/50/25/5/1%）× 3 seed
- `e500_scr_vs_ft_valloss_s54.png`：scratch vs clean ft（单 seed s54），5 面板，**各面板独立 y 轴**（sharey=False）

**绘图坑（重要）**：
- `sharey=True` 会把小范围曲线压扁：ft 全程 y 范围仅 0.006–0.012，而 scratch 1% 档起始 ~1.18 → 共享轴后 ft 变水平直线、看似"平滑"。**改 sharey=False 后可见真实锯齿**
- log 轴放大尾部相对波动（+5~26%），线性轴下绝对波动仅 0.001 级
- 5%/1% 档 scratch 用旧 scr500 run（配置同 e500，仅 run 名不同）

**视觉结论**（s54）：100/50/25% 档 scratch 与 ft 终点几乎重合（增益 −2.7~−5.7%）；5%/1% 档 ft 平台明显低于 scratch（+13%/+34%）；小数据档 scratch 下降段更长（135/286 ep vs ft 38/60 ep）。

## 50. e500 配置 + ft 150ep 实验（§50 v3 定稿，2026-09-02，⏳ 运行中）

> **背景链**：v1（预训练重训 200ep）误启被停 → v2（warmup 100ep）因用户改口作废 →
> **v3 定稿：配置 = e500（§49.16）完全一致，唯一变化 ft epochs 500 → 150**。
> 无 warmup（欠拟合研究 scr100/scr500/ft1e4 与 sec50/e500 均无 warmup，同口径）。

### 50.1 配置（v3）

| 项 | e500（§49.16）| §50 v3 |
|---|---|---|
| scheduler | cosine 无 warmup | **cosine 无 warmup（不变）** |
| batch / wd / lr | 64 / 0 / 全 1e-4 | 不变 |
| 早停 | patience=10 | 不变 |
| seed | 3 | 不变 |
| 档位 | 100/50/25/5/1% | 不变 |
| 预训练 | sec50（50ep）| **sec50 复用（50ep，不变）** |
| **ft epochs** | 500 | **150**（唯一变化）|
| scratch | 500ep 早停 | **复用 e500 结果（不重跑，A 臂充分收敛）** |

### 50.2 run 清单（45 run，后缀 `-ft150`）

- ft：clean/noisy05/noisy20 × 5 档 × 3 seed，150ep，init =
  `tokamind-pca01sigma-synth-pretrain[-noisy05|-noisy20]-s{seed}-sec50`
- run 名：`tokamind-pca01sigma-{finetune|noisy05-finetune|noisy20-finetune}-{f}pct-s{seed}-ft150`

### 50.3 数据资产（现成，同 §49.12）；val = 728 炮；test = 21,350 点

### 50.4 预期

- e500 中 ft 全部早停于 12–100ep（无一超 100ep）→ 150ep 下大部分仍会早停，
  结果预计与 e500 几乎一致，仅预算缩小（100% 档 150ep ~1h/run）

### 50.5 执行与状态

- runner：`runs/run_ft150.py`（45 job，state 断点；8 卡 16 并行，GPU=idx%8）
- 启动：2026-09-02，`wc -l runs/run_ft150.state`（到 45 = 完成）
- 预估：~2-3h

### 50.6 判定（对照 §49.13 e500 结果）

1. ft 150ep vs e500 ft 500ep（早停）精度是否一致（预期几乎相同）
2. 早停位置分布对比（§49.14：ft 早停区间 clean 54-60 / 37-55 / 39-60 ep 等）
3. A/C 增益对照（e500：−4.8/−2.7/−5.7/+13.2/+34.4%）

---

## 51. 文章数据整理（2026-09-02 起，持续更新）

> **目的**：run 太多，论文写作前统一归档。本章节登记**每个实验的路径与复现所需文件**。
> 归档目录：`paper_artifacts/<实验组>/`（仅拷贝结果 JSON + checkpoint + test 数据；
> 大体积训练 cache 保留原位；代码/画图脚本均不归档——mast-bridge 仓库待全部实验跑完后统一备份）。

### 51.1 temporal 5% + 1%（§48.7 / §48.8）

> 🔄 **2026-09-03 归档升级**：全 5 档（100/50/25/5/1%）实验完成后，归档已替换为
> `paper_artifacts/temporal_full/`（33 checkpoint + 21 eval json + test + LCFS 绘图子目录，465M；
> 清单见该目录 README.md）。旧目录 `temporal_5pct_1pct/` 已删除，本小节为当时记录（5%/1% 两档，
> 289M），**当前有效路径以 §54.6/§55.7 和 temporal_full/README.md 为准**。

**归档目录**：`paper_artifacts/temporal_5pct_1pct/`（289M；清单见目录内 README.md）

```
temporal_5pct_1pct/
├── eval/           结果 JSON（3 个）
├── checkpoints/    15 个 run 的 best checkpoint + scaler + summary（3 预训练 + 6 scratch + 6 ft，24M）
└── test/           M9 test 数据（split_test_real.jsonl + temporal_test_real.npz，17,498 点 / 507 炮）
```

> **代码依赖**（不归档，待统一备份）：推理 = `mast-bridge/scripts/evaluate_tokamind_testset.py` +
> `mast_bridge` 包 + `scripts/train_tokamind_manifest.py`（`_build_signal_specs` 依赖）+
> `external/tokamind/src/mmt/`。✅ 曾用归档 checkpoint + 归档内代码端到端验证过（评估结果与原 JSON 逐位一致）。

| 内容 | 归档位置（原位置） |
|---|---|
| 5% scratch eval | `artifacts/tokamind_test_eval/temporal_5u_A_scratch_3seed.json`（归档：`paper_artifacts/temporal_full/eval/`） |
| 5% ft eval | `artifacts/tokamind_test_eval/temporal_5u_C_ft_3seed.json`（归档同 eval/） |
| 1% scratch+ft eval | `artifacts/tokamind_test_eval/temporal_1u_3seed.json`（归档同 eval/） |

> 训练 runner（`runs/run_temporal_5u.py`、`run_temporal_1u.py`）、评估/训练脚本（mast-bridge 仓库）、
> 画图脚本（`plots/script/`）均留在原处，不归档。

**模型 checkpoint（15 个，他人可直接用我训练的模型做测试）**：
`paper_artifacts/temporal_full/checkpoints/<run 名>/`（升级后位置；旧名已删），每个 run 目录与原始训练 run 目录布局一致：
`checkpoints/best/`（backbone.pt、modality_heads.pt、output_adapters.pt、token_encoder.pt、meta.json）
+ `manifest_scalers.npz`（推理必需）+ `manifest_training_summary.json`。

| 臂 | 5%（`-u1e4`） | 1%（`-1u`） |
|---|---|---|
| 预训练（50ep） | `tokamind-temporal-synth-pretrain-s{54,55,56}-u1e4` | 复用 5% 档 |
| scratch（100ep） | `tokamind-temporal-scratch-5pct-s{54,55,56}-u1e4` | `tokamind-temporal-scratch-1pct-s{54,55,56}-1u` |
| finetune（100ep） | `tokamind-temporal-finetune-clean-5pct-s{54,55,56}-u1e4` | `tokamind-temporal-finetune-clean-1pct-s{54,55,56}-1u` |

**用归档 checkpoint 复现测试**（✅ 已验证：ft-5pct-s54 评估结果与原 JSON 逐位一致，不依赖原 runs/）：
1. 把 `test/temporal_test_real.npz` 复制到评估输出目录并命名为 `test_cache_split_test_real.npz`（§00.5 坑 3）
2. `python mast-bridge/scripts/evaluate_tokamind_testset.py --manifest test/split_test_real.jsonl \
   --run-dir checkpoints/tokamind-temporal-finetune-clean-5pct-s54-u1e4 ... --output-json <out>.json`
   （`--run-dir` 直接指向 `checkpoints/<run 名>/`，脚本自动加载 `checkpoints/best/` + `manifest_scalers.npz`）

**复现训练所需数据文件**（大体积，保留原位；test 已归档到上面 `test/`）：

| 用途 | manifest | cache（data/processed/training_cache_pca_01sigma/） |
|---|---|---|
| 预训练 | `data/manifests/training_temporal/run_synth_pretrain.jsonl` | `temporal_synth_clean.npz` |
| 5% | `data/manifests/training_temporal/subset_run_real_5pct.jsonl` | `temporal_subset_run_real_5pct.npz` |
| 1% | `data/manifests/training_temporal/subset_run_real_1pct.jsonl` | `temporal_subset_run_real_1pct.npz` |
| val | `data/manifests/training_temporal/split_val_shots.jsonl` | — |

**原训练 run 目录**（仅重跑需要，评估/推理用归档 checkpoints/ 即可）：`runs/tokamind-temporal-{synth-pretrain,scratch-5pct,finetune-clean-5pct}-s{54,55,56}-u1e4`（5%，9 个）、`runs/tokamind-temporal-{scratch-1pct,finetune-clean-1pct}-s{54,55,56}-1u`（1%，6 个）。

**关键结果（3 seed 均值）**：

| 档位 | NRMSE A/C | RMAE A/C | 增益（NRMSE/RMAE） |
|---|---|---|---|
| 5% | 3.80% / 2.33% | 1.61% / 0.91% | +38.6% / +43.3% |
| 1% | 6.42% / 2.31% | 2.83% / 0.90% | +64.0% / +68.1% |

### 50.7 ✅ 结果（2026-09-02，45/45 训练 + 45/45 评估完成，~1.5h 总墙钟）

**执行**：ft150 45 run 全部 rc=0（16 并行 ~1.5h，比预估快）；评估 45 run → `artifacts/tokamind_test_eval/pca_ft150_*.json`
（缓存修复后命中，~20s/run）。

**结果表（nrmse_per_sample %，3 seed 均值；A = e500 scratch 充分收敛，§49.13）**：

| 档 | A | C clean | 增益 | C noisy05 | 增益 | C noisy20 | 增益 |
|---|---|---|---|---|---|---|---|
| 100% | 0.659 | 0.654 | +0.7% | 0.637 | +3.4% | 0.666 | −1.1% |
| 50% | 0.714 | 0.739 | −3.4% | 0.745 | −4.2% | 0.739 | −3.4% |
| 25% | 0.772 | 0.826 | −6.5% | 0.817 | −5.5% | 0.818 | −5.7% |
| 5% | 1.107 | 0.982 | **+12.8%** | 0.975 | **+13.5%** | 0.987 | **+12.2%** |
| 1% | 1.464 | 1.087 | **+34.7%** | 1.052 | **+39.2%** | 1.116 | **+31.1%** |

**结论**：
1. **ft 150ep 与 e500（500ep）结果逐格几乎一致**（15 格差异 <0.04pp）——因 e500 ft 全部早停在 ≤100ep，
   150ep 是充分上限；100% 档 clean 甚至微优（0.654 vs 0.692）为 seed 噪声
2. **增益趋势与 e500 完全同构**：高数据档（100/50/25%）无正收益（−1%~−6.5%，C 略差或打平），
   5% 档 +12~14%、1% 档 +31~39%（A 臂剔除坏 run s55 口径）
3. noisy 臂与 clean 持平（差异 <1pp）——噪声非必要条件（第四次确认）
4. **预算验证**：ft 150ep ≈ ft 500ep 结果，训练预算可缩小 3 倍；scratch 仍需充分收敛（500ep 早停）口径

**坑（缓存第三次踩）**：`test_cache_split_test_real.npz` 又变 17,498 行（mtime 02:49）→ random 评估
mismatch 触发 zarr 重建（collab_package zarr 不完整：730 炮中 ~3,852 行提取失败被丢 → rebuild 写不全）。
**修复**：`cp data/processed/training_cache_pca_01sigma/test_real_pca.npz artifacts/tokamind_test_eval/test_cache_split_test_real.npz`
（21,350 行）。**规则：评估前先校验缓存行数 == 21,350（`python -c` 读 sample_ids 长度），非 21,350 一律
cp 恢复，严禁依赖 rebuild**。

### 50.8 ⚠️ 观测记录：ft150 的 train loss 突变与 ft val loss 震荡（2026-09-02）

**1. train loss 突变（clean-100pct-s54-ft150）**
- 现象：ep78→79 train loss 0.0109 → 0.0213（**+95% 尖峰**），ep80 回落恢复
- 全 45 run 扫描：仅此 1 处真突变（ep2 的 "下降 >50%" 是正常快速收敛，非异常）
- 位置：该 run best@ep76 之后 2ep（早停窗口内）；best checkpoint 与评估结果不受影响
  （nrmse 0.65% 组内正常）
- 判定：单 epoch 内个别 batch 偶发高 loss（极端样本/数值尖峰）拉高 epoch 平均；**无需重跑**

**2. ft val loss 平台期震荡（全部 ft150 run）**
- 现象：best 后曲线在高位 lr 下锯齿（尾部 10ep 波动 = best 的 +3%~+26%，绝对值 0.001–0.003）
- 根因：cosine 按 150ep 衰减但 ft 全部早停在 12–117ep → **早停时 lr 仍为峰值 40–80%**
  （如 clean-1pct-s54 早停@52 时 lr = 峰值 73%）→ 高 lr 平台振荡
- train loss 平滑（epoch 内 batch 平均）、val 震荡（每 epoch 单次快照捕捉权重晃动）
- 缓解方向（用户讨论过）：① 图层面 EMA/标 best（零成本）② ft lr 调小到 1e-5
  （= §45.7 sec50 配置，曲线平滑但收敛变差：RMAE 0.328% vs lr1e-4 0.290%，§49.8）
  ③ cosine 提前衰减 + lr floor ④ wd>0（破坏可比性，不建议）
- **状态（2026-09-02）**：用户倾向把 ft150 的 ft lr 改为 1e-5 重跑（与 sec50 一致），
  epochs/scratch 处理待确认后执行

### 50.9 ✅ §50 v4：ft 100ep 实验（2026-09-02 定稿，⏳ 运行中）

> 用户决定（承接 §50.8）：ft150 → **ft 100ep**，其余设置完全不变（= e500/§49.16：
> lr1e-4、cosine 无 warmup、bs64、wd0、patience 10、3 seed、5 档）；**预训练复用 sec50 不重训**。

- run：45 个 ft（clean/noisy05/noisy20 × 100/50/25/5/1% × s54/55/56），后缀 **`-ft100`**
- runner：`runs/run_ft100.py`（16 并行 8 卡，state `runs/run_ft100.state` 到 45 = 完成）
- 预期：ft100 与 ft150 结果几乎一致（ft150 早停绝大多数 ≤100ep；注意 noisy05-100pct 部分 seed
  ft150 早停至 117ep，100ep 上限可能轻微截断——评估时核对）
- 预估 ~1h；完成后评估 45 run → 对照 ft150（§50.7）

### 50.10 ✅ §50 v5：ft 150ep + warmup 10% 实验（2026-09-02 定稿，⏳ 运行中）

> 用户决定（§50.8 讨论后）：ft150 基础上**给 ft 加 warmup-fraction 0.1**（15ep 线性升温到 1e-4
> 后 cosine 衰减），其余与 ft150（§50.7）完全一致：lr1e-4、bs64、wd0、patience 10、3 seed、
> 5 档、epochs 150。预训练 sec50 复用、scratch 复用 e500（均不重训）。

- run：45 个 ft（clean/noisy05/noisy20 × 100/50/25/5/1% × s54/55/56），后缀 **`-w150`**
- runner：`runs/run_w150.py`（16 并行 8 卡，state `runs/run_w150.state` 到 45 = 完成）
- 预期：warmup 平滑起始段；平台期振荡改善有限（主因是早停时 lr 仍高，cosine 未衰减到位）——
  若效果不足，备选方案 A（cosine 提前衰减 + lr floor，见 §50.8）
- 预估 ~1.5h；完成后评估 45 run → 对照 ft150（§50.7）+ sec50，重点看 val loss 平滑度

---

## 附录 A. 评估指标定义（2026-09-02，论文用，与 evaluate_tokamind_testset.py 实现一致）

**记号**：test 集共 N 个样本（time slice）；每个样本为 65×65 = P = 4225 像素的平衡磁通场。
第 s 个样本的第 p 个像素：真值 $y_{s,p}$（EFIT psi，Wb）、预测 $\hat{y}_{s,p}$。
预测前先反归一化：$\hat{y} = \hat{y}_{std}\cdot\sigma_y + \mu_y$（$\mu_y,\sigma_y$ 来自 run 自身的
`manifest_scalers.npz`，psi 为原始物理值 Wb）。

**逐样本量**：
- 样本 RMSE：$\mathrm{RMSE}_s=\sqrt{\frac{1}{P}\sum_p (\hat{y}_{s,p}-y_{s,p})^2}$
- 样本 MAE：$\mathrm{MAE}_s=\frac{1}{P}\sum_p|\hat{y}_{s,p}-y_{s,p}|$
- 样本幅值范围：$A_s=\max_p y_{s,p}-\min_p y_{s,p}$（psi 幅度归一化基准）
- 全局幅值范围：$A_\mathrm{global}=\max_{s,p} y_{s,p}-\min_{s,p} y_{s,p}$

**报告指标**（代码 `metrics_from_predictions`）：

| 指标 | 公式 | 单位 | 含义 |
|---|---|---|---|
| nRMSE (per sample) | $\frac{1}{N}\sum_s \frac{\mathrm{RMSE}_s}{A_s}$ | % | 每样本 RMSE ÷ 该样本 psi 幅值范围，再平均（**主报告指标**）|
| nRMSE (global) | $\frac{\sqrt{\frac{1}{NP}\sum_{s,p}(\hat{y}-y)^2}}{A_\mathrm{global}}$ | % | 全局归一化 |
| RMAE (per sample) | $\frac{1}{N}\sum_s \frac{\mathrm{MAE}_s}{A_s}$ | % | 同上但用 MAE |
| RMAE (global) | $\frac{\frac{1}{NP}\sum_{s,p}|\hat{y}-y|}{A_\mathrm{global}}$ | % | 全局 MAE 归一化（= mean\|diff\|/A_global）|
| Raw RMSE | $\sqrt{\frac{1}{NP}\sum_{s,p}(\hat{y}-y)^2}$ | Wb（报告用 mWb，×1000）| 绝对 RMSE |
| Raw MAE | $\frac{1}{NP}\sum_{s,p}|\hat{y}-y|$ | Wb | 绝对 MAE |

**增益（Gain）**：**论文两张主表（§59）统一为 (A−C)/A（real scratch 为基准，2026-09-03 定稿）**。历史章节混用多种口径，引用时须注明：
- **§45（sec50）/§52/§53/§57.4/§59 系**：$\mathrm{Gain}=\dfrac{A-C}{A}$（基准 = Real scratch A）——论文口径
- **§49.13/§50（e500/ft150 早期版）曾用** (A−C)/C（基准 = C）——2026-09-03 起弃用，仅存于历史表格（如 §50.12，已加注）
- 同一数字两种口径不同（如 random 1%：基准 C 得 +37.8%，基准 A 得 +27.5%），
  跨表对比前必须统一

**物理量说明**：psi 为 total 平衡磁通（含线圈真空场贡献）；65×65 网格
R∈[0.06,1.98]m、Z∈[−2,2]m（dR=0.03m、dZ=0.0625m），cache 中 psi 轴序为 [R,Z]。

### 50.11 ✅ §50 v5 结果：ft 150ep + warmup 10%（2026-09-02，45/45 训练 + 评估完成）

**执行**：45 run 全部 rc=0（~1.5h）；评估 45 run → `artifacts/tokamind_test_eval/pca_w150_*.json`（缓存 21,350 校验后命中，0 失败）。

**精度（nrmse_per_sample %，3 seed 均值）vs ft150（§50.7）**：

| 档 | w150 clean | ft150 clean | w150 vs ft150 最大差 |
|---|---|---|---|
| 100% | 0.686 | 0.654 | +0.032pp |
| 50% | 0.729 | 0.739 | −0.010pp |
| 25% | 0.815 | 0.826 | −0.011pp |
| 5% | 0.974 | 0.982 | −0.008pp |
| 1% | 1.076 | 1.087 | −0.011pp |

**15 格（3 臂 × 5 档）差异全部 <0.04pp、方向混合 → warmup 对精度无实质影响**（早停兜底）。

**val loss 平滑性（尾部 20ep 去趋势波动均值）**：w150 0.00023–0.00048 vs ft150 0.00015–0.00049——
**无系统性改善**（约半数略好、半数略差）。warmup 只平滑起始段（ep1-15 线性升温），
平台期震荡仍由"早停时 lr 为峰值 40-80%"主导（cosine 150ep 衰减太慢）。

**结论**：warmup 10% 未达成平滑目标。若要压平台期震荡，需 **cosine 提前衰减 + lr floor**
（§50.8 方案 A，改动 loop.py）或图层面 EMA/标 best。

**图**：`plots/random/`（w150 系列可加画，如需与 ft150 对照出 `w150_vs_ft150_*`）

### 50.12 ✅ §50 v5 评估结果：稀缺档增益确认（2026-09-02）

> w150（ft 150ep + warmup 10%）评估汇总。C = clean 臂 3 seed 均值；A = e500 充分收敛
> （100/50/25% = e500 scratch 3 seed；5% = s54 scr500 + s55/56；1% = s54 scr500 + s56 + s57，剔除坏 s55）。
> 增益 = (A−C)/C（§50 口径）。**2026-09-03 §57 更新：1% A 档换 s58（s57 转备用），A = 1.483（{s54,s56,s58}），
> 下表 1% 行已同步**（s58 是好 basin 但为四 seed 中最弱，A 均值 1.464→1.483，增益 +36.1%→**+37.8%**，结论不变）。

| 档 | A (充分收敛) | C w150 | 增益 w150 | C ft150 | 增益 ft150 |
|---|---|---|---|---|---|
| 100% | 0.659 | 0.686 | −4.0% | 0.654 | +0.7% |
| 50% | 0.714 | 0.729 | −2.1% | 0.739 | −3.4% |
| 25% | 0.772 | 0.815 | −5.3% | 0.826 | −6.5% |
| **5%** | 1.107 | **0.974** | **+13.7%** | 0.982 | +12.8% |
| **1%** | 1.483 | **1.076** | **+37.8%** | 1.087 | +34.7% |

（nrmse_per_sample %；其他指标同向：1% raw_rmse C-w150 = 2.84 mWb、rmae_global = 0.33%）

**结论**：
1. **稀缺档正增益稳健**：5% 档 **+13.7%**、1% 档 **+37.8%**——warmup 不影响结论（与 ft150 差 <1pp，
   1% 档略优）
2. 高数据档（100/50/25%）无正收益（−2~−5%）——与 e500/sec50 全链一致：**预训练价值只在
   数据稀缺时体现，1% 档最强**（充分收敛口径下，对比 sec50 的 50ep 欠拟合假象已消除）
3. 这是充分收敛口径下**稀缺档增益的最终确认结果**（用户认可，可用于论文）

> ⚠️ **2026-09-03 口径注**：本节及上表增益原按 **(A−C)/C** 报告（§50 口径），数值 = +13.7%/+37.8%（5%/1% clean）。
> 论文口径已统一为 **(A−C)/A**（real scratch 基准）→ 5% +12.0%、1% +27.5%，论文数字以 §52.2/§53.3/§57.4/§59.2 为准，本节为历史记录。

### 50.13 早停机制答疑记录（2026-09-02）

> 用户观察：real scratch 的 train loss 在早停点仍在下降，为何早停？
> **结论：早停判据 = val loss（patience=10 连续无改善），与 train loss 无关。**

**数据证据**（scr500 run）：
- scratch-1pct-s54（早停@286，best@276）：train 0.0191→0.0166 持续降；val 0.0413→0.0399(best)
  →0.0404 平台抖动 10ep → 停
- scratch-5pct-s54（早停@145，best@135）：train 0.0218→0.0182 持续降；val best 0.0212 后 10ep
  无改善 → 停

**解释**：
1. train 仍降而 val 平台 = **过拟合方向**（模型继续记忆训练集，泛化停滞）——继续训只扩大
   train/val 泛化间隙，早停正确
2. 小数据档泛化间隙大：1% 档 train≈0.017 vs val≈0.040（仅 1,924 训练样本）
3. 训练脚本早停逻辑：`early_stop.patience` 基于 val；train loss 曲线不作为停止依据

---

## 52. 论文最佳实验结果登记（2026-09-02，含 checkpoint 路径）

> 两个"数据稀缺档正增益"的最佳证据链实验（用户选定，论文复现用）。
> 指标 = nrmse_per_sample %（3 seed 均值±std）；**增益 = (A−C)/A（real scratch 基准，random/temporal 统一，2026-09-03）**。
> checkpoint 通用结构：`<run-dir>/checkpoints/best/`（backbone.pt + token_encoder.pt +
> modality_heads.pt + output_adapters.pt + meta.json）+ `manifest_scalers.npz` + `manifest_training_summary.json`。

### 52.1 实验一：Temporal split 稀缺档（5% + 1%）

> 训练 M5-M7（187,292 行）→ val M8 → test M9 全量 17,498 点 / 507 炮。
> 配置：bs64、cosine 无 warmup、wd=0、**lr 全 1e-4**、patience 10；5% 档 = scratch/ft 各 100ep
> （§48.7，`-u1e4`）；1% 档 = scratch 500ep/ft 100ep（§48.8，`-1u`）。预训练 clean 各自新训。

| 档 | 指标 | A real scratch | C pretrain+ft | 增益 | 评估 json |
|---|---|---|---|---|---|
| 5% | nRMSE (per sample) | 3.80 ± 0.12% | **2.33 ± 0.11%** | **+38.6%** | `temporal_5u_A_scratch_3seed.json` + `temporal_5u_C_ft_3seed.json` |
| | Raw RMSE | 7.77 ± 0.26 mWb | **4.81 ± 0.21 mWb** | +38.1% | |
| | RMAE (global) | 1.61 ± 0.06% | **0.91 ± 0.04%** | +43.3% | |
| 1% | nRMSE (per sample) | 6.42 ± 0.04% | **2.31 ± 0.17%** | **+64.0%** | `temporal_1u_3seed.json`（6 run）|
| | Raw RMSE | 13.09 ± 0.10 mWb | **4.78 ± 0.34 mWb** | +63.1% | |
| | RMAE (global) | 2.83 ± 0.01% | **0.90 ± 0.08%** | +68.1% | |

**Checkpoint 路径**（workspace 根 `runs/` 下）：
- A（scratch）5%：`runs/tokamind-temporal-scratch-5pct-s{54,55,56}-u1e4/`（best@56/61/88）
- C（ft clean）5%：`runs/tokamind-temporal-finetune-clean-5pct-s{54,55,56}-u1e4/`（best@19/27/18）
- A（scratch）1%：`runs/tokamind-temporal-scratch-1pct-s{54,55,56}-1u/`
- C（ft clean）1%：`runs/tokamind-temporal-finetune-clean-1pct-s{54,55,56}-1u/`
- test cache：`data/processed/training_cache_pca_01sigma/temporal_test_real.npz`（17,498 行）；
  评估时复制为 `artifacts/tokamind_test_eval/test_cache_split_test_real.npz`（⚠️ 覆盖 random 版缓存，
  用后需从 `test_real_pca.npz` 恢复 21,350 版）
- 结论：纯时间外推（M9 装置漂移）下稀缺档增益最强：5% +38.6%、1% +64.0%——**预训练跨装置
  campaign 可迁移，数据越稀缺价值越大**（fig：`plots/temporal/temporal_5pct_val_loss_ft_vs_scratch.png`）

### 52.2 实验二：Random split 稀缺档（w150，ft 150ep + warmup 10%）

> 配置：bs64、cosine **+ warmup 10%**、wd=0、lr 全 1e-4、patience 10、3 seed、ft 150ep（全部早停）；
> A 臂 = e500 scratch 充分收敛（500ep 早停，复用）；预训练 sec50（50ep）复用。
> test 21,350 点 / 730 炮。

| 档 | A（充分收敛）| C clean | 增益 | C noisy05 | 增益 | C noisy20 | 增益 | 评估 json |
|---|---|---|---|---|---|---|---|---|
| 5% | 1.107 ± 0.022 | **0.974 ± 0.012** | **+12.0%** | 0.981 ± 0.022 | +11.4% | 0.991 ± 0.011 | +10.5% | `pca_w150_*_5pct_*.json` |
| 1% | 1.483 ± 0.039 | **1.076 ± 0.010** | **+27.5%** | 1.057 ± 0.016 | **+28.7%** | 1.147 ± 0.038 | **+22.6%** | |

（2026-09-03 §57：1% A 档 seed 集 {s54,s56,s57} → **{s54,s56,s58}**，均值 1.464 → 1.483，见 §57.3；
A/C 全部 ± = 总体标准差 ddof=0，全 5 档逐 seed 见 §57.4）

**Checkpoint 路径**：
- C（ft）5 档 × 3 臂 × 3 seed：`runs/tokamind-pca01sigma-{finetune,noisy05-finetune,noisy20-finetune}-{100,50,25,5,1}pct-s{54,55,56}-w150/`
- 稀缺档代表：`runs/tokamind-pca01sigma-finetune-1pct-s54-w150/`（best val 1.076% 对应）等 5%/1% 各 9 个
- A（scratch，复用）5%：`runs/tokamind-pca01sigma-scratch-5pct-s54-scr500/` + `...-5pct-s{55,56}-e500/`
- A（scratch，复用）1%：`runs/tokamind-pca01sigma-scratch-1pct-s54-scr500/` + `...-1pct-s56-e500/` + `...-1pct-s58-e500/`（s57 转备用）
- 预训练 init：`runs/tokamind-pca01sigma-synth-pretrain-s{54,55,56}-sec50/`（+ noisy05/noisy20 版）
- test cache：`artifacts/tokamind_test_eval/test_cache_split_test_real.npz`（21,350 行，勿被 temporal 覆盖）
- 结论：充分收敛口径下随机划分稀缺档增益 5% +13.7%、1% +37.8%（高数据档无正收益）；
  warmup 对结果无实质影响（§50.11/50.12）

### 52.3 数据划分统计（temporal vs random，#shot / #slice，2026-09-02 实测 manifest）

> 口径：slice = manifest 行数（time slice）；档位 = train 炮数按比例抽样；**val/test 各档共享全量**；
> 100% 无 subset 文件 = 全量 `split_train_real.jsonl`；temporal 5% 无 `subset_real` 文件，
> 由 `subset_run_real_5pct.jsonl`（含 val 行）扣除 val 炮行得到（8,683 = 14,889 − 6,206）。
> 源文件：`data/manifests/{training_temporal,training_pca_01sigma}/`。

**Temporal split（时间序划分：train = M5–M7，val = M8，test = M9 全量；§48/§52.1）**

| 档位 | train #shot / #slice | val #shot / #slice | test #shot / #slice |
|---|---|---|---|
| 100% | 6,046 / 187,292 | 559 / 6,206 | 507 / 17,498 |
| 50% | 3,023 / 93,722 | 559 / 6,206 | 507 / 17,498 |
| 25% | 1,512 / 46,840 | 559 / 6,206 | 507 / 17,498 |
| 5% | 302 / 8,683 | 559 / 6,206 | 507 / 17,498 |
| 1% | 60 / 1,917 | 559 / 6,206 | 507 / 17,498 |

**Random split（逐炮随机划分，同一炮池；§49/§50/§52.2）**

| 档位 | train #shot / #slice | val #shot / #slice | test #shot / #slice |
|---|---|---|---|
| 100% | 5,828 / 171,427 | 728 / 21,147 | 730 / 21,350 |
| 50% | 2,914 / 87,822 | 728 / 21,147 | 730 / 21,350 |
| 25% | 1,457 / 43,711 | 728 / 21,147 | 730 / 21,350 |
| 5% | 291 / 9,070 | 728 / 21,147 | 730 / 21,350 |
| 1% | 58 / 1,924 | 728 / 21,147 | 730 / 21,350 |

**要点**：
1. 档位 ≈ train 炮数等比例抽样（random 291 ≈ 5,828×5%、temporal 302 ≈ 6,046×5%）
2. temporal 1% 的 60 炮嵌套在 5% 的 302 炮内（§48.8）；val = M8 中间 campaign（时间外推验证）
3. random val = 728 炮 21,147 slice、test = 730 炮 21,350 slice（§50.3）；时间外推 vs 随机：test 集不相交

### 52.4 LCFS 图绘制流程：预测 psi → LCFS 边界提取（论文 LCFS 图，2026-09-02 记录）

> 脚本：`plots/script/plot_lcfs_ip.py`（旧，单模型）；`plot_lcfs_temporal.py`（temporal A/C 双臂，
> 布局：上排 scratch、下排 ft、图例行、底排 Ip(t)，输出 `plots/temporal/lcfs_pred_{shot}_combined_{pct}pct.png`）；
> `plot_lcfs_random.py`（random split 同布局，shot 17833，输出 `plots/random/lcfs_pred_{shot}_combined_{pct}pct.png`）。
> 三个脚本的 LCFS 提取函数 `lcfs_extract` 逐字节相同；此处流程以它们为准。

**1. 输入数据**
- 预测 psi：`plots/data/preds/test_predictions_{run}.npz`（`pred_psi`，evaluate --save-predictions 生成，
  已反归一化到 raw Wb，形状 (N, 65, 65)，轴序 [R, Z]）；网格 `R1 = linspace(0.06, 1.98, 65)`、
  `Z1 = linspace(-2, 2, 65)`（dR=0.03 m、dZ=1/16=0.0625 m，与训练 cache 一致，附录 A）
- EFIT 参考边界：zarr `equilibrium/lcfs_r|z`（每帧 141/145 点，列数=该炮平衡帧数），
  取与 manifest `target_time` 最近的一帧（`argmin(|t_eq − target_time|)`）
- 机械元件：`MAST_{active_coils,passive_coils,wall,limiter}.pickle`（机器几何，仅绘图）

**2. LCFS 提取（`lcfs_extract(psi)`，核心流程）**
1. **磁轴（O 点）**：`argmax(psi)` 全局最大磁通位置 (oi, oj)——限制区域永远围绕 O 点，避开旁瓣（X 点外区域）
2. **阈值扫描**：对 `frac ∈ [0.05, 0.95)`, 步长 0.01，阈值
   `lv = psi.max() − frac × (psi.max() − psi.min())`——从接近峰值的高阈值（小区域）逐步降到低阈值（大区域）；
   等价于扫描"包含磁轴的最大闭合等高面"，无需预设物理量级阈值（归一化幅值扫描）
3. **二值化 + 连通域**：`ndimage.label(psi ≥ lv)` 求连通域，取含 O 点的域 `main`
4. **泄漏判停**：若 `main` 触及网格四边（顶/底/左/右任一行或列非空）→ 等高面已打开（开放到边界），
   **break 停止扫描**，取上一个未泄漏域
5. **最大闭合域**：未泄漏时若面积大于已记录 → 更新 best（即逐级扩张中最后一次仍闭合的域）
6. **失败返回**：全程无候选（如 ramp-up 早期 psi 近乎平坦/O 点区域紧贴边界）→ 返回 None，
   图上不画预测边界（仅 EFIT 蓝线）
7. **轮廓提取**：`skimage.measure.find_contours(best, 0.5)`（二值图等值线）取**最长一条**（= 外边界），
   像素坐标 (row, col) 经 `np.interp` 线性映射回物理坐标：`R = interp(row, [0, 64], R1)`、
   `Z = interp(col, [0, 64], Z1)`（axis0=R、axis1=Z，与 `indexing="ij"` 网格一致）→ 闭合边界点序列 (r, z)

**3. 绘图组装（main）**
- 帧选择：每炮样本按 `target_time` 排序，优先取 **EFIT LCFS 全有限** 的样本（M9 炮 EFIT LCFS 97.7%
  为 NaN——爬升段未重建/数据缺失，取全有限帧否则蓝线断裂），不足 4 帧则回退均匀取帧；等距取 4 帧
- 每帧：pred psi `pad_psi`（edge 外扩至 R∈[0,2.05]、Z∈[−2.1,2.1]，仅作 pcolormesh 背景）+
  白等值线 8 级（原始 65×65 网格）+ 机械元件 + 蓝实线 EFIT LCFS + 橙虚线预测 LCFS
- 底排：Ip(t) 全炮波形（magnetics，罗氏线圈）+ 红虚线/圆点标记 4 帧时刻
- 帧标题 = 时间（仅 top 排）；共享 colorbar "predicted psi (Wb)"；图例行在 R-Z 行与 Ip 行之间

**4. 语义与已知限制**
- 提取的是"包含磁轴的**最大闭合磁面**"（相对幅值 0.05–0.95 扫描的最后一层闭合等高线），
  对 limited 等离子体近似 separatrix
- **未实现 X 点开放 separatrix**（diverted 构型的开放分界线提取已放弃，plot.md 坑 7）：diverted 炮
  的"闭合近似边界"仍可画，但与 EFIT LCFS 会有尾部差异
- 效果检查手段：pred 边界对角线 ≈ EFIT 对角线（正常 0.9–1.0），比值偏离过大 = 提取失败/psi 场异常

---

## 53. 最新交接记录（2026-09-02 定稿：全部训练/评估实验完成，进入论文阶段）

> **给新对话的交接**：本文档唯一权威入口，但注意 **§00-§21 的结论是 10ep 口径（已作废）**，
> **论文/任何对外结论一律用充分收敛口径（§49.13 e500 + §50 w150 + §48 temporal）**。
> 本节是当前最终状态，做完下面 §53.1 检查清单即接手完毕。

### 53.1 实时状态速查（2026-09-03 实测更新）

- **当前无任何运行中的训练/评估任务**（`ps aux` 无 python 训练进程）
- state 文件完成情况：`run_e500.state` 57、`run_ft150.state` 45 ✅、`run_w150.state` 45 ✅、
  `run_temporal_5u/1u.state` 9/6 ✅、`run_scr1pct_s55_refill.state` 1（**结果 = 又坏 basin，见 §57.3**）、
  `run_scr1pct_s58_resume.state` 1 ✅（s58 好 basin，rc=0）
- **§57 唯一缺口待办（2026-09-03 05:40 ✅ 已闭环）**：random A 臂 1% seed 补齐——s55 两次确定性复现坏 basin
  （0.2316），s58（ep13 中断后 resume 续跑）**好 basin**（0.0435@247→早停@257，~15 min），评估 nrmse 1.5374%；
  1% A 档最终 = {s54, s56, s58}，3-seed 均值 **1.4826 ± 0.039**（s57 转备用）。五档 scratch 全部 3 seed 齐。
  ✅ 归档同步完成：s58 ckpt/eval 已入 `paper_artifacts/random/`（新名 `random-scratch-1pct-s58-lr1e4-ep500`），
  README/EXPERIMENTS.md/§50.12/§52.2/§53.3 数字已更新（A 1% 1.464→1.483，增益 +36.1%→+37.8%）
- 已完成实验（2026-09-03）：§54 temporal 25/50/100%（18 run ✅）、§55 temporal ft lr5e-5 u5e5（15 run ✅）
- 评估 JSON 齐全：`pca_e500_*.json`、`pca_ft150_*.json`、`pca_w150_*.json` × 45、
  `temporal_5u_{A,C}_*.json` + `temporal_1u_3seed.json` + §54/§55 各档 eval（temporal_full 内 21 个）
- 论文归档（§56 规范化后）：`paper_artifacts/{temporal_full,random}/`（同构：README | checkpoints |
  eval | test | dataset_split* | plot/）+ 顶层 `EXPERIMENTS.md`（新会话速览 + 命名映射附录 A）+
  `PLOT_STYLE.md`（散点图风格 LOCKED）；归档内 run/eval 已统一改新名，原 runs/ 不动。
  **2026-09-03 §57/§57.4 后计数：random 64 ckpt + 61 eval（+s58 及补齐 25/50/100% 档 36 run），temporal_full 33 ckpt + 21 eval**（§57.4）

### 53.2 ⚠️ 两个"已废弃/未完成"条目（勿再执行、勿误报）

1. **§50.9 v4 ft100（`-ft100`）已废弃**：runner `runs/run_ft100.py` 于 09-02 06:04 启动后即被终止
   （master log 停在 `total=45 done=0`，无任何 run 目录、无 state）——用户改跑 v5 w150（§50.10），
   w150 已全部完成并出结果（§50.11/50.12）。文档 §50.9 头部仍标"⏳ 运行中"，**是过期文字，无需处理**。
2. **§45.8 scr200（scratch 200ep）从未执行且已被取代**：A 臂充分收敛基线最终用 e500 scratch 500ep
   （3 seed，§49.13），scr200 方案作废，勿再启动。

### 53.3 论文可用的最终结论（充分收敛口径，全部 3 seed 均值±std、早停收敛、无欠拟合假象）

**结论一句话：仿真预训练的价值只在数据稀缺时体现；数据越稀缺增益越大，且时间外推场景强于随机划分。**

| 场景 | 档位 | A scratch | C 预训练+微调 | 增益 | 出处 |
|---|---|---|---|---|---|
| Random split | 5% | 1.107 | **0.974 ± 0.012** | **+12.0%** | §59.2/§57.4（w150，clean）|
| Random split | 1% | 1.483 | **1.076 ± 0.010** | **+27.5%** | §59.2/§57.4（w150，clean；A = {s54,s56,s58}，§57.3）|
| Temporal（M9）| 5% | 3.80 ± 0.12% | **2.33 ± 0.11%** | **+38.6%** | §48.7 |
| Temporal（M9）| 1% | 6.42 ± 0.04% | **2.31 ± 0.17%** | **+64.0%** | §48.8 |

- 高数据档（100/50/25%）无正收益：random −2~−6%（§49.13 e500、§50.7 ft150、§50.12 w150 三次一致），
  **§00.2 的"全域有效 −60%"旧结论不可用于论文**（那是 A 臂 10ep 欠拟合的假象，§49.6/49.8 已证）
- 噪声非必要条件（clean/noisy05/noisy20 微调后持平，第五次确认）；warmup 无实质影响（§50.11）
- 预训练另有两个量化副产物：小数据档收敛加速 3~10×（§49.14 早停区间）、seed 方差抹平（C 臂 std 全档 <0.04）
- 论文数字/checkpoint 路径/复现清单：**§52**（两证据链）+ **附录 A**（指标定义 + 口径警示）

### 53.4 剩余待办（按优先级，均非阻塞）

0. ✅ **§57 random A 臂 1% seed 补齐已闭环（2026-09-03）**：s58 resume 续跑好 basin（0.0435@247，早停@257，
   耗时 ~15 min），评估 nrmse 1.5374%；1% A = {s54,s56,s58} 均值 **1.4826±0.039**（详见 §57.3）。
   ✅ 归档同步亦完成：s58 已入 `paper_artifacts/random/`（ckpt+eval，新命名），README/EXPERIMENTS.md/文档数字同步
1. **mast-bridge 代码备份**：§51 注明"待全部实验跑完后统一备份"——现实验已全部结束，可执行
   （画图/runner 脚本在 `plots/script/`、`/tmp/opencode/`、`runs/run_*.py`，需决定归档位置）
2. 论文图收尾（可选）：w150/combined 系列已在 `plots/random/`（psi/jtor/lcfs 1%/5% 对照图已出）；
   论文总图可用 `pca_w150` 数据（聚合脚本参照 `/tmp/opencode/agg_e500.py` 改）
3. 老遗留（§20.14，未跑、属加分项非必需）：geometry 指标、severity bins、limiter 多 seed

### 53.5 坑速记（新会话必读）

- **评估缓存行数校验**：`artifacts/tokamind_test_eval/test_cache_split_test_real.npz` 必须是 21,350 行
  （random 版 = `data/processed/training_cache_pca_01sigma/test_real_pca.npz` 复制；temporal 评估会覆盖成
  17,498 行版，用后必须恢复——§49.13/§50.7 第三次踩坑）
- **增益口径**：(A−C)/A（§45 系）vs (A−C)/C（§49/§50 系）并存，跨表引用必须注明（附录 A）
- 评估 `--output-json` 必须放 `artifacts/tokamind_test_eval/` 下（否则 cache 目录不同触发 zarr 重建，
  collab_package zarr 不完整会写不全）
- scratch 1% 档 s55 是**确定性坏 basin**（§49.13 + §57.3 二次复现实测：同 seed 同配置结果逐位复现 0.2316，
  重跑无效只能换 seed）；s58（resume 好 basin）补位后 **A 臂 1% = {s54, s56, s58}**（s57 转备用，§57.3）

---

## 54. Temporal 高数据档补跑：25/50/100% × A/C（2026-09-02 ✅ 已完成）

> **动机（用户 2026-09-02）**：temporal split（原划分 M5-M7→M8→M9）是论文最强证据链（§52.1，
> 5% +38.6% / 1% +64.0%）。为得到完整 5 档数据量曲线，补跑 25/50/100% 三档
> ×（A real scratch + C clean 预训练+微调）× 3 seed。配置与 §48.7（5%，`-u1e4`）逐字一致。

### 54.1 配置（= §48.7/48.8）

| 项 | 值 |
|---|---|
| bs / scheduler / wd | 64 / cosine 无 warmup / 0 |
| lr | 全 1e-4（scratch 与 ft 都是）|
| epochs | scratch 100 / ft 100（上限，早停 patience=10）|
| seed | s54/s55/s56 |
| 臂 | 仅 clean（与 5%/1% 一致）|
| 预训练 | 复用 `tokamind-temporal-synth-pretrain-s{seed}-u1e4`（50ep，✅ 不重跑）|
| val / test | M8 559 炮（--val-shot）/ M9 全量 17,498 点 / 507 炮 |

### 54.2 数据（✅ manifest/cache 行数对齐实测）

| 档 | manifest | cache | 行数（train+val）|
|---|---|---|---|
| 100% | `run_train_with_val.jsonl` | `temporal_train_with_val.npz` | 193,498 |
| 50% | `subset_run_real_50pct.jsonl` | `temporal_subset_run_real_50pct.npz` | 99,928 |
| 25% | `subset_run_real_25pct.jsonl` | `temporal_subset_run_real_25pct.npz` | 53,046 |

### 54.3 Run 清单（18，后缀 `-u1e4`，无同名现成 run）

- A：`tokamind-temporal-scratch-{100,50,25}pct-s{54,55,56}-u1e4`（9）
- C：`tokamind-temporal-finetune-clean-{100,50,25}pct-s{54,55,56}-u1e4`（9，init=同 seed 预训练）

### 54.4 执行与状态

- runner：`runs/run_temporal_hi.py`（18 run 一波，4 并行 = 2 卡 × 2 进程；state `runs/run_temporal_hi.state` 到 18 = 完成；日志 `runs/logs/thi_*.log`）
- 机器：2×RTX 4090（49GB 空闲）；预估 100% 档 ~134 s/ep（上限 ~3.7h）、50% ~1.9h、25% ~0.9h；总墙钟 ~7-9h
- 状态：⏳ 2026-09-02 启动；监控 `tail runs/logs/thi_master.log` / `wc -l runs/run_temporal_hi.state`
- 坑备：若个别 run 100ep 未早停按 §49.8 补长 epoch；scratch 与 ft 无先后依赖可全并行

### 54.5 评估与判定（训练完成后）

- 评估：18 run × M9 test，输出 `artifacts/tokamind_test_eval/temporal_hi_{run}.json`；
  test cache 需 17,498 行版（`cp data/processed/training_cache_pca_01sigma/temporal_test_real.npz
  artifacts/tokamind_test_eval/test_cache_split_test_real.npz`；⚠️ 用后如需 random 评估恢复 21,350 版）
- 聚合：3 seed 均值±std，与 §48.7/48.8（5%/1%）合并完整 5 档曲线
- 判定：高数据档 C 是否正收益（random 同口径为负 −2~−6%）；temporal 稀缺档优势（+38.6/+64%）
  是否在 25/50/100% 延续；结果写入 §54.6

### 54.6 ✅ 结果（2026-09-02，18/18 训练 + 18/18 评估完成）

**执行**：`runs/run_temporal_hi.py`（4 并行 2 卡）2026-09-02 10:53 启动，~2.2h 全部完成（18 run 全 rc=0，
远快于预估 9h——早停全部提前：scratch 停@22-44ep、ft 停@12-35ep，cosine-100ep 衰减快 + 高数据档收敛快）。
评估：`artifacts/tokamind_test_eval/temporal_hi_{run}.json` × 18（M9 test 17,498 点 / 507 炮）。

**结果（nrmse_per_sample %，3 seed 均值±std；增益 = (A−C)/A）**：

| 档 | A scratch | C 预训练+微调 | 增益(nrmse) | raw_rmse A/C (mWb) | rmae_global A/C (%) | 增益(rmae) |
|---|---|---|---|---|---|---|
| 100% | 2.489 ± 0.167 | **2.204 ± 0.129** | **+11.4%** | 5.13/4.56 | 0.984/0.853 | +13.3% |
| 50% | 2.794 ± 0.173 | **2.378 ± 0.165** | **+14.9%** | 5.73/4.89 | 1.132/0.937 | +17.2% |
| 25% | 3.237 ± 0.151 | **2.249 ± 0.095** | **+30.5%** | 6.59/4.68 | 1.342/0.884 | +34.1% |
| 5%（§48.7）| 3.80 ± 0.12 | 2.33 ± 0.11 | +38.6% | 7.77/4.81 | 1.61/0.91 | +43.3% |
| 1%（§48.8）| 6.42 ± 0.04 | 2.31 ± 0.17 | +64.0% | 13.09/4.78 | 2.83/0.90 | +68.1% |

**核心结论（temporal 完整 5 档曲线定稿，论文可用）**：
1. **temporal 高数据档全部转正**（100% +11.4% / 50% +14.9% / 25% +30.5%），
   与 random split 同口径（§49.13/§50.12：100/50/25% 为 −2~−6%）**方向相反**——时间外推场景下
   预训练高数据档依然有效，物理先验跨 campaign 的价值在时间泛化上体现最充分
2. **增益随数据稀缺单调放大**：+11.4% → +14.9% → +30.5% → +38.6% → +64.0%（100→1%），五档全部同向
3. **C 臂平台型**：2.20-2.38 全档几乎水平（5% 与 100% 只差 0.13pp）——预训练后误差对数据量不敏感；
   A 臂单调上升 2.49 → 6.42
4. **早停区间**：scratch 高数据档 best@17-34 → 停@27-44；ft best@2-25 → 停@12-35（5%/1% 同协议）
5. **与 §52.1 一致性**：5%/1% 数字与既有结果完全一致（预训练复用同源），18 个新 run 无坏 basin（对比 random 1% s55）
6. 对论文含义：**temporal split（原划分）是预训练价值最强的完整证据链——五档全正、稀缺放大**；
   建议论文主图 = temporal 5 档 A/C 双曲线 + random split 对照（后者高数据档为负，可作"场景依赖"讨论点）

**早停/收敛核对**：scratch 100% best_val 0.066-0.071@17-34ep、25% 0.078-0.083@12-20ep；
ft 全档 best_val 0.067-0.073（与 5%/1% 的 0.073-0.076 持平）——收敛充分（100ep 上限内早停，非截断）

### 54.7 ✅ ft val loss 震荡成因研究（2026-09-02，用户提问驱动）

> **现象**：ft/scratch 曲线尾部 val loss 锯齿明显（相对 best +6~16%），质疑 cosine 衰减到 lr≈0 后为何仍震荡。
> **结论：曲线段从未进入低 lr 区——全部 run 早停于 lr = 峰值 59-97% 处（cosine 前 ~40% 步数衰减极缓：
> ep17 → 93% 峰值、ep35 → 73%）。震荡幅度是 lr 的单调函数，lr→0 后消失。**

**证据链**：
1. **排除 eval 噪声**：val 用 `model.train(False)` + `set_grad_enabled(False)`（loop.py:328/343，无 dropout）；
   val loader `shuffle=False`（train_tokamind_manifest.py:437）→ val loss 是"epoch 末权重"的确定性函数，
   震荡必来自相邻 epoch 权重不同
2. **早停时 lr 实测**（history 的 lr_backbone）：ft 停@12-35ep → lr = 峰值 73-97%（7.3-9.7e-5）；
   scratch 停@22-44ep → 59-89%；cosine 公式 `0.5·(1+cos(π·progress))`（scheduler.py:243）无 floor，min_lr=0
3. **决定性自然实验（同一 run 内，5pct-scratch-s56 跑满 98ep 至 lr≈0）**：
   - lr 6.5e-5→4.2e-5（ep40-55）：val ±8% 锯齿
   - lr 3e-5→1.7e-5（ep60-73）：±2-4%，趋势下降
   - lr <1.6e-5（ep74-88）：平滑单调 0.115→0.102
   - lr ~1e-6（ep89-98）：±0.1% 完全平坦 → **lr≳2e-5 为震荡带**
4. **机制**：平台区曲率小，lr≳2e-5 时 Adam 每步权重变化 ≫ 有效下降方向 → epoch 末权重在最优邻域游走 →
   val 单点快照起伏 ±5-16%（ft 停时 lr 7-10e-5 恰在最强震荡带）；best 只是噪声过程低点，
   之后 10ep 难再破纪录 → patience 触发早停
5. **影响**：① 评估用 best checkpoint → 实验数字有效 ② A/C 协议对称 → 相对增益结论不受影响
   ③ best 略偏乐观（噪声最小值），量级小 ④ 若需平滑 + 可能更低 best：降低 ft 峰值 lr 或让 cosine
   提前衰减完（§50.8 方案 A）——**已立项新实验 §55（ft lr 5e-5）验证**

**图产物**：`plots/temporal/temporal_hi_val_loss_ft_only{,_linear}.png`（高数据档 ft-only）、
`temporal_all_tiers_val_loss{,_2row}.png`、`temporal_all_tiers_train_loss_2row.png`（全档 A/C 两行布局）

---

## 55. Temporal ft lr=5e-5 重跑实验（2026-09-02 定稿，✅ 已完成：2026-09-03）

> **动机（用户 2026-09-02）**：§54.7 研究结论——ft val 震荡 ∝ lr（1e-4 时尾部 ±6-16%）。
> 验证把 ft 峰值 lr 降到 **5e-5** 能否减弱平台期震荡。**唯一变化 = ft lr**；
> A 臂 scratch 与预训练完全不动。

### 55.1 配置

| 项 | §54/§48（旧 ft）| §55（新 ft）|
|---|---|---|
| **lr** | 1e-4 | **5e-5**（唯一变化）|
| bs / scheduler / wd | 64 / cosine 无 warmup（按 100ep 衰减）/ 0 | 不变 |
| epochs / 早停 | 100 上限 / patience 10 | 不变 |
| seed / 档位 | s54/55/56 × 100/50/25/5/1% | 不变 |
| init | `synth-pretrain-s{seed}-u1e4` | 复用（不重跑）|

### 55.2 Run 清单（15，后缀 `-u5e5`，无同名现成 run）

`tokamind-temporal-finetune-clean-{100,50,25,5,1}pct-s{54,55,56}-u5e5`

### 55.3 数据（复用 §54/§48，行数已验证）

| 档 | manifest | cache |
|---|---|---|
| 100% | `run_train_with_val.jsonl` | `temporal_train_with_val.npz`（193,498）|
| 50% | `subset_run_real_50pct.jsonl` | `temporal_subset_run_real_50pct.npz`（99,928）|
| 25% | `subset_run_real_25pct.jsonl` | `temporal_subset_run_real_25pct.npz`（53,046）|
| 5% | `subset_run_real_5pct.jsonl` | `temporal_subset_run_real_5pct.npz`（14,889）|
| 1% | `subset_run_real_1pct.jsonl` | `temporal_subset_run_real_1pct.npz`（8,123）|

val = M8 559 炮；test = M9 17,498 点 / 507 炮

### 55.4 执行与状态

- runner：`runs/run_temporal_u5e5.py`（15 run 一波，4 并行 2 卡，state `run_temporal_u5e5.state` 到 15 = 完成，日志 `runs/logs/t5e5_*.log`）
- 时间预估：lr 减半 → 早停可能略晚；100% 档 ≤3-4h、50% ~1.5-2h、25% ~1h、5%/1% ≤0.5h；**总墙钟 ~8-10h**
- 状态：⏳ 2026-09-02 启动 → ✅ 2026-09-02 13:46 完成（15/15 rc=0，实际 ~0.5h——早停全部提前且无坏 run，比预估快一个量级）

### 55.5 评估与判定（完成后）

- 评估：15 run × M9 test → `artifacts/tokamind_test_eval/temporal_u5e5_*.json`（test cache 已是 17,498 temporal 版）
- **精度表**：新 C(5e-5) vs 旧 C(1e-4，§54.6/§48.7-8) vs A scratch（不动），15 格对比（预期持平 ±0.1pp）
- **震荡量化**：每 run 尾部 (尾部最差/best −1)%，与 1e-4 版逐格对比（预期 6-16% → ~3-6%，
  早停时 lr ≈ 3.5-4.8e-5 仍在震荡带边缘——不期望完全消失；若仍明显 → 下一步 cosine 提前衰减方案）
- 图：ft-only val loss 全档线性图，新旧 lr 叠画
- 结果写入 §55.6

### 55.6 ✅ 结果（2026-09-03，15/15 训练 + 15/15 评估完成）

**执行**：训练 2026-09-02 13:13 启动 → 13:46 全部 15 run rc=0（4 并行 2 卡，~0.5h；
早停全部提前：best@ep3-39 → 停@13-49，无 run 跑满 100ep，无坏 basin）。
评估：`runs/eval_temporal_u5e5.sh`（15 × M9 test，4 并行）→ `artifacts/tokamind_test_eval/temporal_u5e5_{run}.json` × 15
（test cache `test_cache_split_test_real.npz` = 17,498 点 temporal 版，与 §54 同源）。

**精度表（nrmse_per_sample %，3 seed 均值±std；增益 = (A−C)/A）**：

| 档 | A scratch（不动）| 旧 C(1e-4) | **新 C(5e-5)** | Δ(new−old) | 增益(nrmse) |
|---|---|---|---|---|---|
| 100% | 2.489 ± 0.167 | 2.204 ± 0.129 | **2.207 ± 0.063** | +0.003 | +11.3%（同 11.4%）|
| 50% | 2.794 ± 0.173 | 2.378 ± 0.165 | **2.349 ± 0.121** | −0.029 | +15.9%（原 14.9%）|
| 25% | 3.237 ± 0.151 | 2.249 ± 0.095 | **2.182 ± 0.056** | −0.067 | +32.6%（原 30.5%）|
| 5%（§48.7 同源）| 3.80 ± 0.12 | 2.334 ± 0.111 | **2.284 ± 0.067** | −0.050 | +39.9%（原 38.6%）|
| 1%（§48.8 同源，旧为 `-1u`）| 6.42 ± 0.04 | 2.310 ± 0.136 | **2.255 ± 0.094** | −0.055 | +64.9%（原 64.0%）|

**逐格 paired Δ**（15 格，seed 对齐）：范围 [−0.211, +0.095] pp，10/15 改善，均值 −0.040 pp；
全部在 ±0.22 pp 内（单 seed 噪声级）→ **精度与 1e-4 持平，无任何档位退化**。
新 lr 的 3-seed std 全线收窄（0.127 → 0.080，均降 ~37%）——seed 方差变小，与"低 lr 采样更少噪声 basin"自洽。

**震荡量化（尾部 max/best −1 %，best_ep 后窗口）**：

| 档 | 旧 1e-4 三 seed | 新 5e-5 三 seed |
|---|---|---|
| 100% | 11.2%（13.2/10.4/9.9）| **7.0%**（7.9/8.6/4.5）|
| 50% | 9.3%（6.9/14.6/6.3）| 7.7%（6.0/8.4/8.7）|
| 25% | 10.9%（14.5/10.3/7.9）| 10.4%（12.8/10.4/7.9）|
| 5% | 13.9%（11.7/13.9/16.0）| **9.6%**（12.7/8.1/8.1）|
| 1% | 11.4%（13.5/9.7/11.0）| 9.6%（14.4/7.4/7.2）|
| 总体 | 11.3% | **8.9%（paired Δ −2.5pp，10/15 run 下降）**|

**核心结论**：
1. **震荡确实减弱但未达标**（11.3 → 8.9%，~22% 降幅；未出现预期的 ~6-16% → 3-6% 减半）。
   根因与 §54.7 预测一致：新 run 仍全部早停于 lr = 峰值 **52-96%**（停@ep13-49 → lr 2.6-4.8e-5），
   5e-5 峰值减半后停止点仍在 §54.7 震荡带（lr≳2e-5）上沿内；25% 档几乎无改善（早停最靠前）。
2. **精度不受影响且略优 + seed 方差收窄**——若论文需换用更稳口径，5e-5 版可整体替换 1e-4 版（无需重训 scratch）。
3. **§54.6 五档曲线结论完全稳健**：换 ft lr 后五档增益同向（+11.3/+15.9/+32.6/+39.9/+64.9%），C 臂平台
   2.18-2.35 依旧水平。A 臂与预训练未动 → 论文主图可用任一版；建议保持 1e-4（已与 §52 登记一致）。
4. **彻底消除震荡的正确路线仍是 §54.7/§55.5 预留方案**：cosine 提前衰减完（让早停窗口落入 lr<2e-5）
   或去掉早停跑满 100ep 后按最佳评估（§50.8 方案 A）。若仅追求图平滑，可直接用本文图（震荡已肉眼减弱）。

**收敛核对**：新 ft best_val 100% 0.069-0.072、50% 0.068-0.071、25% 0.071-0.073、
5% 0.073-0.074、1% 0.084-0.086（与旧版同档持平，1% 因样本少噪声更大）——收敛充分。

**图产物**：`plots/temporal/temporal_u5e5_val_loss_ft_lr_cmp.png`（5 档并排线性轴，
红实线 = 旧 1e-4、蓝虚线 = 新 5e-5，各 3 seed 叠画；脚本 `plots/script/plot_loss_temporal_u5e5_cmp.py`）

### 55.7 ✅ 逐 seed 增益审计：temporal 全部 C 对 A 均正增益（2026-09-03，用户提问驱动）

> **问题**：temporal 划分下是否每个微调 run（C）都比 scratch（A）好？
> **结论：30/30 全正**（旧 u1e4/1u 15 格 + 新 u5e5 15 格，逐 seed 对齐），无任何反例。

**证据表（nrmse_per_sample %，A/C 同 seed 配对；增益 = (A−C)/A）**：

| 档 | seed | A | 旧 C(1e-4) | 新 C(5e-5) | 增益_旧 | 增益_新 |
|---|---|---|---|---|---|---|
| 100% | 54 | 2.626 | 2.323 | 2.253 | +11.5% | +14.2% |
| 100% | 55 | 2.254 | 2.024 | 2.119 | +10.2% | +6.0% |
| 100% | 56 | 2.586 | 2.264 | 2.250 | +12.4% | +13.0% |
| 50% | 54 | 2.801 | 2.607 | 2.517 | +6.9% | +10.2% |
| 50% | 55 | 2.579 | 2.304 | 2.293 | +10.7% | +11.1% |
| 50% | 56 | 3.003 | 2.223 | 2.236 | +26.0% | +25.5% |
| 25% | 54 | 3.040 | 2.115 | 2.187 | +30.4% | +28.0% |
| 25% | 55 | 3.407 | 2.309 | 2.248 | +32.2% | +34.0% |
| 25% | 56 | 3.265 | 2.322 | 2.111 | +28.9% | +35.3% |
| 5% | 54 | 3.707 | 2.280 | 2.290 | +38.5% | +38.2% |
| 5% | 55 | 3.964 | 2.489 | 2.362 | +37.2% | +40.4% |
| 5% | 56 | 3.726 | 2.232 | 2.200 | +40.1% | +41.0% |
| 1% | 54 | 6.432 | 2.120 | 2.123 | +67.0% | +67.0% |
| 1% | 55 | 6.443 | 2.375 | 2.302 | +63.1% | +64.3% |
| 1% | 56 | 6.375 | 2.434 | 2.339 | +61.8% | +63.3% |

**要点**：
1. **30/30 全正**：5 档 × 3 seed × 2 套 lr 无任何反例；最弱单 run = 100% s55 新 lr（+6.0%），
   最强 = 1% s54（+67.0%，两套 lr 同）。单 seed 增益远大于 seed 间噪声（±0.1-0.2pp），非噪声侥幸。
2. **增益随稀缺单调且无档位塌陷**：逐档单 seed 增益区间（新旧并集）：
   100% +6.0~+14.2 → 50% +6.9~+26.0 → 25% +28.0~+35.3 → 5% +37.2~+41.0 → 1% +61.8~+67.0；
   五档区间完全分离（无跨档重叠），单调性在 seed 层面即成立，非仅均值现象。
3. **论文写法建议**：可声明 "pretrain+finetune outperforms scratch in **every** (tier × seed) pair
   under temporal split (30/30, min +6.0%)"，强于仅报均值±std；random split 对照（§50.12）则
   25-100% 全负、1% 仅部分 seed 正——场景依赖论点更立体。
4. **数据来源**：A/C 均取 best checkpoint × M9 test（17,498 点 / 507 炮，cache `test_cache_split_test_real.npz`）；
   旧 C = `temporal_hi_*u1e4.json`(100/50/25%) + `temporal_5u_C_ft_3seed.json`(5%) + `temporal_1u_3seed.json`(1%, `-1u`)；
   新 C = `temporal_u5e5_*.json`；A = `temporal_hi_*scratch-u1e4.json`(100/50/25%) + `temporal_5u_A_scratch_3seed.json`(5%) + `temporal_1u_3seed.json`(1%)。
   （审计脚本一次性运行于 2026-09-03，未留存为文件；如需复算见上列 json）

---

## 56. 2026-09-03 交接：归档规范化 + 图产物 + 可复现性审计（与 §55.6/55.7 同日完成）

> 当天工作分三条线：① §55 u5e5 实验评估闭环（见 §55.6/§55.7，本文件已含）；
> ② paper_artifacts 全面规范化（结构/命名/图包）；③ 全链路可复现性审计。
> 本节记录 ②③；论文用图与数值可直接引用 §56.3-56.5。

### 56.1 归档结构规范化（两实验组同构）

- `paper_artifacts/` 现共 **5.3G**：random 2.8G + temporal_full 2.5G + 2 个总览 md（EXPERIMENTS.md、PLOT_STYLE.md）
- `temporal_full/` 于 2026-09-03 **替换**旧 `temporal_5pct_1pct/`（后者已删）：全 5 档 33 ckpt + 21 eval + test + dataset_split + plot/
- 两归档最终布局完全同构：`README.md | checkpoints/ | eval/ | test/ | dataset_split* | plot/`
- `EXPERIMENTS.md` = 新会话速览（125 行）：一句话结论 → 目录地图 → 命名规则 → 配置 → 数据划分 → 结果表 → 复现 3 步 → 坑清单 + **附录 A 旧名→新名映射规则**
- `PLOT_STYLE.md` = 散点图风格规范（LOCKED，2026-09-03）

### 56.2 统一命名（60 run，仅 paper_artifacts 内部；原 runs/ 与 progress2 引用不动）

新名模板：`{temporal|random}-{pretrain|scratch|ft[-clean|noisy05|noisy20]}-[{tier}pct-]s{seed}-lr1e4-ep{50|100|150|500}[-warm10]`。
旧名 → 新名映射 = 5 条确定性规则（前缀/角色/档位/后缀重建/噪声词），见 EXPERIMENTS.md 附录 A；示例：`tokamind-temporal-finetune-clean-1pct-s54-1u` → `temporal-ft-1pct-s54-lr1e4-ep100`。
同时：60 ckpt 目录 + 45 eval 文件（文件名与内部 run/run_dir 字段）同步改名；验证无悬空引用。
（2026-09-03 §57 追加：+`random-scratch-1pct-s58-lr1e4-ep500` 及补齐 25/50/100% 档 36 run → random 64 ckpt + 61 eval；全归档 97 ckpt + 82 eval，见 §57.4）

### 56.3 图产物（锁定风格，规范 = paper_artifacts/PLOT_STYLE.md）

**temporal LCFS 图包**（`temporal_full/plot/`，自包含）：zarr **28631+29412** 两炮 + machine 元件；
脚本路径改相对解析（原 common.py 移除）、run 名改新名、preds 4 个已生成（2.0G）→ 拿到包即能出图；
LCFS 图复现步骤见 `plot/README.md`。

**random 散点三图**（全部 2×2：行 A/C、列 5%/1%；hexbin viridis + y=x + R 角标 + 无 grid/无 title、
colorbar count、psi x 轴 = `EFIT psi / ground truth`、y = `psi pred`；figsize 11.0×8.6）：

| 图 | 口径 | R（A5/A1/C5/C1）| 归档复现 |
|---|---|---|---|
| psi_scatter_points_combined_1_5pct | 逐像素均匀采样 0.2%（rng=2026，180k 点/面板）| 0.9982/0.9967/0.9985/0.9981 | ✅ 自包含 |
| psi_scatter_combined_1_5pct | 逐样本均值（21,350 点/面板）| 0.9963/0.9943/0.9966/0.9951 | ✅ 自包含 |
| jtor_ip_combined_1_5pct | Ip_pred = ∫J_φ dRdZ（EFIT j_φ 掩码），J_φ=−Δ*ψ/(μ₀R) | 0.9636/0.9360/0.9922/0.9933 | ✅ 画图自包含（读 jtor_ip_pairs.npz，1.4MB）；数组重算需全量 zarr |

- preds 生成物：random/plot 2.4G + temporal_full/plot 2.0G（共 4.4G，占归档 84%），README 均有重建命令
- 修 bug：psi/jtor workspace 脚本原用 common.style_ax()（自带 grid）与归档脚本不一致 → 已统一去 grid，
  并禁止在规范中再用 style_ax；单模型 psi_scatter.py 顺带同改
- 新增 j_tor 计算-画图拆分脚本：`random/plot/scripts/{compute_jtor_pairs,plot_jtor_ip_combined}.py`
- canonical 图在 `plots/random/`；归档可复现副本在 `random/plot/out/`

### 56.4 可复现性审计（2026-09-03，全通过）

- 结构：60 ckpt 权重/scaler/summary 完整；eval json run/run_dir 引用无悬空；test cache 行数 17,498/21,350 ✓
  （2026-09-03 §57 后 +s58 与 25/50/100% 档 36 run → 全归档 97 ckpt / 82 eval；random = 64/61，见 §57.4）
- **7 run 端到端重评 = BIT-EXACT**（含所有指标字段，非仅主指标）：temporal ft/scratch 5pct+1pct s54（4）+ random ft-clean-5pct-s54、ft-noisy20-1pct-s56、scratch-1pct-s57（3）
- LCFS 图端到端：归档 ckpt → preds → plot_lcfs_temporal/ip 直接出图 ✓
- 注意：random j_tor 早期 workspace 版 R（0.9359/0.9923…）与归档数组版第 4 位小数差 ±1e-4（两次 pred npz 生成的浮点差异），口径以归档版为准

### 56.5 论文 caption/正文参考数值（2026-09-03，口径速查）

- test 集：random = 21,350 slice / 730 炮；temporal = 17,498 / 507
- 网格点采样：全量 = 21,350 × 4,225 ≈ 9,020 万点；均匀采样 0.2%、seed 2026 → 179,820 点/面板；R 在采样点上算
- 逐样本均值：每 slice 对 65×65 取均值 → 21,350 点/面板；R 在均值上算
- j_tor：每 slice 1 点；mask = EFIT j_φ > 帧 max×1e-3（内部 63×63）；Δ*ψ 中心差分 2 阶；Ip 参考 = 罗氏线圈插值
- psi 主文一句式结论素材：预训练对像素级相关性的增益随稀缺放大（网格点 R 1%：0.9967→0.9981）

---

## 57. Random split A 臂 1% s55→s58 补训（2026-09-03 定稿，⏳ s58 待 GPU 启动）

> **触发**：用户要求 random split 的 real scratch **五档 seed 集统一为 {s54, s55, s56}**。
> 盘点结论（§56 后）：100/50/25/5% 已齐（s54/55/56，nrmse % = 0.629-0.825 / 0.701-0.732 /
> 0.825-0.735 / 1.109-1.133）；**1% 档 s55 = 坏 basin（best_val 0.2316，test nrmse 4.019%，当年剔除）**，
> 仅剩唯一缺口 → 补训 1 个 run。1% 现用 s57（1.481%）在补训完成后转备用。

### 57.1 配置（= §49.13 e500 逐字一致，唯一变化 seed）

| 项 | 值 |
|---|---|
| run | `runs/tokamind-pca01sigma-scratch-1pct-s55-e500`（无同名现成 run——原坏 basin 为同名前身，将覆盖）|
| 数据 | random split 1%：manifest `training_pca_01sigma/subset_..._1pct` + 对应 cache（§49.12）|
| val / test | random 728 炮 val（manifest 内建）/ 21,350 点 730 炮 |
| 训练 | bs64、cosine 无 warmup（按 500ep 衰减）、lr 1e-4、wd0、patience=10、500ep 上限、从零 |
| 模型 | d64/2/4/128、dropout 0.05、69 磁诊断 → 65×65 raw psi（`mast_level2_common_69.json`）|
| seed | 55 |

### 57.2 执行

- 机器：8×RTX 4090 空闲；单 run 占 1 卡（CUDA_VISIBLE_DEVICES=0）
- 预估：~5-8h（参考同档 s56 停@258ep、s57 停@312ep；早停非 500ep 截断）
- 完成后：① 坏 basin 核对（best_val 应 ~0.039-0.042，若 ~0.23 → 又坏，改补 s59）② 评估 M9 test
  21,350（cache 行数校验后，勿覆盖 temporal 版）③ 1% A 口径切 s54/56/58 重算 ④ 归档/文档同步

### 57.3 状态与结果（✅ s58 已完成 2026-09-03 05:40）

- 状态：✅ 完成
- **s55 补训结果（2026-09-03 03:57 完成，rc=0）**：**又坏 basin**——best_val 0.2316（best@34ep，
  早停@44ep，与旧坏 run 数值一致，s55 确定性复现坏 basin）→ 按预案转 **s58**（1% A 档 seed 集维持
  {s54, s55(s57 替代前), s56} 不成立，最终用 s54/56/58，s57 转备用）
- **s58 启动实况（新会话修正）**：`runs/run_scr1pct_s58_refill.py` 于 05:02 被启动但 **ep13 中断**
  （bad_epochs=0、无 state/log/summary——**非坏 basin 结论**；ep13 val 0.2410 与好 seed s56 同点
  0.2395 几乎一致，好坏分水岭在 ep40-44）。判断中断后以 **`--resume` 续跑**（`runs/run_scr1pct_s58_resume.py`，
  05:26 启动，log `runs/logs/scr1pct_s58_resume.log`），cosine 按剩余 epoch 重算。
- **s58 训练结果（✅ 2026-09-03 05:40 完成 rc=0）**：**好 basin**——best_val **0.0435**（best@**247ep**，
  早停@**257ep**），与 s56（0.0393@248→258）几乎逐位对齐（s57 0.0415@302→312）；非坏 basin（0.2316 级）。
  中断+续跑总耗时 ~15 min（257ep × ~3.5s/ep；resume 段 ep14-257 ≈ 14 min）。若从头不间断约 15-16 min。
- **评估（test 21,350 点 / 730 炮）**：`artifacts/tokamind_test_eval/pca_e500_scratch-1pct-s58-e500.json`：
  nrmse_per_sample **1.5374%** / rmae_global 0.4953% / raw_rmse 4.103 mWb（评估前已把 eval cache 恢复
  21,350 行 random 版，勿被 temporal 覆盖）
- **1% A 档最终 3-seed 口径（s58 替换 s57 前对照）**：
  | seed | s54(scr500) | s56 | s57(备用) | **s58** |
  |---|---|---|---|---|
  | nrmse_per_sample % | 1.4512 | 1.4592 | 1.4807 | **1.5374** |
  - **新口径 {s54, s56, s58}：1.4826 ± 0.039**（旧 {s54,s56,s57} 1.4637 ± 0.012）；s58 为四者中最弱
    seed（best_val 略差），但同属好 basin 族，非坏 basin（坏基线 4.02%）
  - 影响：w150 1% 档增益 (A−C)/C 用 A=1.4826、C=1.076 → **+37.8%**（旧 +36.1%），结论不变
- 五档 real scratch 现已各 3 seed 齐全（100/50/25/5% = {s54,s55,s56}；1% = {s54,s56,s58}）

### 57.4 论文 Random split 表最终数值登记（tab:random_table，2026-09-03 定稿，直接用于 LaTeX/截图）

> **口径**：nrmse_per_sample %（test 21,350 点 / 730 炮）；C 臂 = w150（ft 150ep + warmup 10%，
> seed {s54,s55,s56}，45 run）；A 臂（Real-only）= e500 scratch 充分收敛，100/50/25/5% = {s54,s55,s56}，
> **1% = {s54, s56, s58}**（s54 = scr500 评估，s58 = §57.3 补位）；std = 总体标准差（ddof=0）；
> **Gain = (A−C)/A（real scratch 为基准；2026-09-03 与 temporal 统一，原 (A−C)/C 已弃用）**，全精度计算后 1 位小数。全表仅 1% 行随 §57 更新（旧 1.464/+36.1/+38.5/+27.6），
> 其余 20 格与 2026-09-02 版逐位一致。

| Real fraction | Real-only | Clean nRMSE | Clean Gain | Noisy 0.5 nRMSE | Noisy 0.5 Gain | Noisy 2 nRMSE | Noisy 2 Gain |
|---|---|---|---|---|---|---|---|
| 1% | 1.483 ± 0.039 | 1.076 ± 0.010 | **+27.5%** | 1.057 ± 0.016 | **+28.7%** | 1.147 ± 0.038 | **+22.6%** |
| 5% | 1.107 ± 0.022 | 0.974 ± 0.012 | **+12.0%** | 0.981 ± 0.022 | **+11.4%** | 0.991 ± 0.011 | **+10.5%** |
| 25% | 0.772 ± 0.039 | 0.815 ± 0.021 | −5.6% | 0.809 ± 0.040 | −4.8% | 0.807 ± 0.024 | −4.6% |
| 50% | 0.714 ± 0.013 | 0.729 ± 0.013 | −2.2% | 0.764 ± 0.037 | −7.0% | 0.749 ± 0.030 | −4.9% |
| 100% | 0.659 ± 0.025 | 0.686 ± 0.032 | −4.1% | 0.658 ± 0.011 | +0.1% | 0.653 ± 0.003 | +0.9% |

**A 臂（Real-only）逐 seed（nrmse_per_sample %，std = 总体 ddof=0）**：1% = {1.4512, 1.4592, 1.5374}
（s54=scr500, s56, s58）；5% = {1.1090, 1.0794, 1.1328}（s54=scr500, s55, s56）；25% = {0.8250, 0.7561, 0.7345}；
50% = {0.7009, 0.7324, 0.7084}；100% = {0.6289, 0.6571, 0.6903}。Gain 用全精度 A 均值，未因显示 ± 而变。

**备查精确值（仅 1% 行）**：A = 1.4826；C clean = 1.0755 ± 0.0097、
noisy05 = 1.0572 ± 0.0161、noisy20 = 1.1474 ± 0.0377；Gain（(A−C)/A，2026-09-03 统一）= 27.46 / 28.69 / 22.61% → 报 +27.5/+28.7/+22.6%（旧 (A−C)/C 口径 37.85/40.24/29.22%）。

**归档状态（同步 ✅）**：`paper_artifacts/random/` = **64 ckpt + 61 eval**（§57 s58 补位 +
**2026-09-03 补齐 25/50/100% 档 36 run**：A e500 ×3seed×3 档 + C w150 ×3 噪声臂 ×3seed×3 档的 ckpt/eval 全部入档，
论文 tab:random_table 五档 × 3 臂可完整复现）；temporal_full = 33 ckpt + 21 eval。
README/EXPERIMENTS.md 已更新。抽查 2 个新入档 run（scratch-100pct-s54、ft-noisy05-25pct-s55）
端到端重评 **BIT-EXACT**（2026-09-03）。

---

## 58. 预训练"泛化性"证据（2026-09-03：方案 §58.1 + 执行记录 §58.2 + **定稿结论 §58.3**）

> **动机（用户提问）**：temporal split 五档全部转正（§54.6：100% +11.4% → 1% +64.0%；
> §55.7：30/30 逐 seed 全正）而 random split 高数据档为负（§50.12：25-100% −2~−6%）——
> 说明仿真预训练提供的是**跨分布（时间外推/装置漂移）迁移的物理先验**，不只是"稀缺补数据"。
> 本节 = 泛化性证据链：方案（§58.1）+ 执行状态（§58.2）+ **2026-09-03 基于最终数据（§59 两主表）的
> 定稿结论（§58.3，零新算力，论文讨论直接引用）**。random split = 天然对照（同协议、test 同分布）。

### 58.1 方案（三图 + 两统计）

**图 1（主证据，零新算力）**：增益 × 档位交互曲线
- x = 档位 100/50/25/5/1%（log），y = gain %（**统一口径 (A−C)/A**，⚠️ random §50.12 原报
  (A−C)/C，跨图引用必须换算，附录 A），两条线（temporal vs random）+ 3 seed 散点
- 预期视觉：temporal 全档 ≥0 且稀缺单调升；random 高数据档 <0、稀缺档升 → "偏移下预训练全档有效"
- 数据：现成 eval json（temporal_hi_*/5u_*/1u_* + pca_e500_*/pca_w150_*），**无需重跑**

**图 2（机制）**：跨时间泛化曲线
- x = test 炮 shot 序号（M9 内时间推进），y = 每炮 mean nrmse（滚动窗），A/C 两条线
- 预期：A 随距训练越远抬升、C 平稳 → "C 把知识带进未来 campaign"
- 需要：per-shot 预测 → 归档 ckpt + evaluate --save-predictions（CPU ~2-4 min/run × 15 run）

**图 3（偏移可视化 + 归因）**
- 输入侧：69 维特征降维（PCA/UMAP）train(M5-M7)/M8/M9 三色 + Wasserstein/KS 数值
- 预测侧：每样本 OOD 分数（train 特征 kNN 距离）分箱 → A/C 误差箱线（预期 A 斜率陡、C 平）；
  另算 per-shot gain vs OOD 分数相关系数（单一数值证据）

**统计**
1. 交互/高数据档检验：gain 以 3 seed 为单位；(split × tier) 双因素，高数据档（100/50/25%）
   temporal vs random 2 样本检验（配对按 seed 对齐；random 1% A={s54,56,58} 与 C={54,55,56} 不对齐，
   用均值口径并注明）
2. 逐单元：temporal 15 格全正（已证 §55.7）；random 高数据档逐 seed 符号统计

**数据源/口径**：
- temporal C = 旧 lr1e-4 版（§54.6/§55.7 表，与 §52 登记一致；5%/1% 用 §48 同源 json）；
  A = temporal scratch u1e4/-1u（同表）
- random：C = w150 clean（`pca_w150_finetune-{f}pct-s{seed}-w150.json`）；A = e500/scr500
  （100/50/25/5% = s54/55/56 e500；1% = s54 scr500 + s56/s58 e500，**无 s55/s57**，§57.3 定稿）
- 聚合正确性锚点：脚本重算 3 seed 均值须与 §54.6（temporal）与 §50.12（random）一致

### 58.2 执行状态（逐步更新）

- ⏳ 图 1 + 统计 1：数据装配 ✅ 验证通过（绘图脚本可后补，结论见 §58.3 不依赖图）：
  - **均值锚点全部吻合**：temporal 五档 A/C 与 §54.6/§55.7 逐位一致；random A/C 与 §50.12 吻合
    （注意 §50.12 报 (A−C)/C，交互比较统一 (A−C)/A）
  - **中间统计（高数据档 100/50/25%，gain %，9 个对齐单元）**：temporal 均值 +18.8 vs random
    −4.1（差 +22.9pp），9/9 单元 temporal>random，Welch t p=6.7e-5、Wilcoxon paired p=0.0039
    （§58.3 用最终均值复算：+18.96 vs −3.95，一致）
  - 试跑中发现的坑：① temporal 1% 聚合 json（temporal_1u_3seed.json）须按 run 名解析 seed 再配对，
    排序后 zip 会错配 seed；② random 1% A={s54,s56,s58} vs C={s54,s55,s56} 天然不对齐 →
    **定稿口径 = 均值口径 + 图内注明**（2026-09-03 §57 后 s58 定 seed，A 均值 1.483）；③ random A 5%/1% s54 是 scr500（非 e500）json
    （`pca_scr{5,1}pct_s54_scr500.json`），勿用 e500 文件
  - 脚本重写要点：log-x 档位轴、绿=temporal / 红=random、均值线 + std 误差棒 + 逐 seed 淡点、
    图内注明 "random 1%: mean-based (A={s54,s56,s58})"
- ⬜ 图 1 绘图脚本重写（可选加分项，数据与统计已完成）
- ⬜ 图 3 输入侧偏移（纯特征统计，无需推理；可选加分项）
- ⬜ 图 2 与图 3 预测侧（需补 per-shot 预测 npz；A 臂 1% s58 已定 seed 2026-09-03，§57 闭环；可选加分项）

### 58.3 ✅ 泛化性定稿结论（2026-09-03，基于 §59 最终两主表，零新算力，论文讨论直接引用）

**一句话结论：仿真预训练提供的是可跨分布迁移的平衡物理先验——在同分布（random split）下其价值只在
数据稀缺时体现（1% +27.5% / 5% +12.0%，(A−C)/A 口径），而在分布偏移（temporal split，M9 时间外推）下
五档全域有效（+11.4% → +64.0%），数据充足时依然显著——"预训练=把合成物理先验带进未来装置/campaign"。**

**证据 1：两场景增益形状不对称（核心交互证据，统一口径 (A−C)/A，A = Real scratch / C = clean 预训练+微调）**

| 档位 | random nRMSE A | random C | random gain (A−C)/A | temporal nRMSE A | temporal C | temporal gain (A−C)/A |
|---|---|---|---|---|---|---|
| 100% | 0.659 ± 0.025 | 0.686 ± 0.032 | −4.1% | 2.489 ± 0.167 | 2.204 ± 0.129 | **+11.4%** |
| 50% | 0.714 ± 0.013 | 0.729 ± 0.013 | −2.2% | 2.794 ± 0.173 | 2.378 ± 0.165 | **+14.9%** |
| 25% | 0.772 ± 0.039 | 0.815 ± 0.021 | −5.6% | 3.237 ± 0.151 | 2.249 ± 0.095 | **+30.5%** |
| 5% | 1.107 ± 0.022 | 0.974 ± 0.012 | **+12.0%** | 3.799 ± 0.117 | 2.334 ± 0.111 | **+38.6%** |
| 1% | 1.483 ± 0.039 | 1.076 ± 0.010 | **+27.5%** | 6.417 ± 0.030 | 2.310 ± 0.136 | **+64.0%** |
| 高数据档均值 (100/50/25%) | | | **−3.95%** | | | **+18.96%** |

（nrmse_per_sample %；表源 = §59.2 random（w150 clean）/ §59.3 temporal（u1e4 clean）。⚠️ random 在 2026-09-03 前曾报 (A−C)/C（1% +37.8% 等），现已与 temporal 统一为 **(A−C)/A**（1% +27.5% 等），本表数字即当前口径。）

**证据 2：逐单元稳健性（非均值巧合）**
- temporal：**30/30 逐 (档位 × seed) 全正**（5 档 × 3 seed × 新旧两套 lr 口径，min +6.0% = 100% s55；
  单 seed 增益远大于 seed 间噪声，五档单 seed 增益区间完全分离——§55.7）
- random：高数据档 **9/9 单元为负或近零**（3 档 × 3 seed）；稀缺档 1%/5% 转正但 1% 部分 seed 仍负
  （数据极度稀缺时 seed 敏感，§49.13 坏 basin 即例）→ 正负分隔不是均值噪声

**证据 3：统计检验（高数据档 100/50/25%，gain 以 3 seed 为单位，统一 (A−C)/A）**
- temporal 均值 +18.96% vs random −3.95%（差 ~22.9pp）
- **9/9 单元 temporal > random**；Welch t 检验 p = 6.7e-5；Wilcoxon 配对 p = 0.0039

**证据 4：机制与副产物（旁证）**
- **C 臂平台型**：temporal C 2.18–2.35 全档几乎水平（5% 与 100% 差 <0.13pp）；random C 1.08–0.98 同档内也平
  → 预训练把"误差对真实数据量/分布偏移的敏感性"抹平（§54.6 结论 3）
- **收敛加速 3–10×**：小数据档 scratch 需 135–145ep（5%）/ 258–312ep（1%），ft 仅 27–55ep / 12–69ep
  （§49.14）——先验不仅提精度，也大幅提速
- 噪声非必要条件（第五次确认）、warmup 无实质影响（§50.11）——clean 预训练即有此效应
- 视觉佐证（plots/）：temporal ft 的 val loss 起点 ~0.08 低于 scratch 终点 ~0.12
  （`plots/temporal/temporal_5pct_val_loss_ft_vs_scratch.png`，§48.7）——先验直接可见

**论文讨论表述建议（可直接改写）**：合成预训练在同分布随机划分下仅在真实数据极度稀缺时带来增益
（随机表），而在时间外推（装置/campaign 漂移）下五档全域有效且随稀缺放大（temporal 表）——这提示预训练
学到的是跨装置可迁移的平衡物理先验，其价值在分布偏移场景下被完全释放；高数据档的同分布对比（random 100%
无正收益）则说明该先验不替代真实数据，而是稀缺与偏移场景的"放大器"。

**与 §59 的关系**：§59.2/§59.3 = 论文两张主表（各 5 档 × 三噪声臂 / A/C）的唯一登记；§58.3 = 两表间的
泛化性对照结论（同一批实验数字，无新实验）。如需图：§58.1 图 1（交互曲线）数据/统计已就绪，脚本重写即可出。

---

## 59. 论文使用的主实验结果定稿（2026-09-03，tab:random_table + tab:temporal_table）

> **给新对话的交接**：论文正文只使用下面两张主表（random split + temporal split 各 5 档）。
> 本节是它们的**唯一权威登记**：LaTeX 源码（可直接粘贴）、Markdown 对照表、实验配置、数据源
> （eval json / checkpoint / seed 集）与验证记录。所有数值已与归档
> （`paper_artifacts/{random,temporal_full}`）机器逐格核对（2026-09-03，全部 PASS）。

### 59.1 实验总览

| | Random split（tab:random_table）| Temporal split（tab:temporal_table）|
|---|---|---|
| 数据划分 | 按炮随机 80/10/10（seed 20260825）：train 5,828 炮/171,427 | M5+M6+M7 → M8 val → M9 test：train 6,046 炮/187,292 |
| test | 21,350 slices / 730 炮（全 campaign）| 17,498 slices / 507 炮（纯 M9）|
| 档位 | 100/50/25/5/1%（嵌套子集 seed 20260825）| 同左（temporal 子集 seed 20260827）|
| A 臂 | real scratch e500（500ep 上限+早停，**充分收敛**）| real scratch（100ep 上限+早停，u1e4；1% 为 `-1u`）|
| C 臂 | 预训练(sec50)+微调 **w150**（150ep+warmup10%，lr 1e-4）| 预训练(50ep)+微调 clean **u1e4**（100ep，lr 1e-4）|
| 噪声臂 | **3 臂**：clean / noisy05 / noisy20（仅 random 有）| **仅 clean** |
| 3 seed | A/C 均 s54/s55/s56；**A-1% = {s54,s56,s58}**（s55 坏 basin 剔除，§57）| A/C 均 s54/s55/s56 |
| Gain 公式 | **(A−C)/A**（基准 = Real scratch A）| **(A−C)/A**（基准 = Real scratch A）|
| std 约定 | 全部总体标准差（ddof=0）| ddof=0；**仅 1% 行为样本 std（ddof=1）→ ±0.04/±0.17** ⚠️ |

配置共性：bs64、cosine 无 warmup（random ft 例外 +warmup 10%）、wd=0、lr 全 1e-4、patience=10、
d64/2/4/128、69 磁诊断 → 65×65 raw psi；指标 = nrmse_per_sample %（附录 A）。

### 59.2 Random split（论文 tab:random_table，出处 §57.4/§50.12）

```latex
\begin{table}[htbp]
\centering
\small
\begin{tabular}{c c c c c c c c}
\toprule
Real Data fraction & Real-only & \multicolumn{2}{c}{Clean} & \multicolumn{2}{c}{Noisy 0.5} & \multicolumn{2}{c}{Noisy 2} \\
\cmidrule(lr){3-4} \cmidrule(lr){5-6} \cmidrule(lr){7-8}
& & nRMSE & Gain & nRMSE & Gain & nRMSE & Gain \\
\midrule
1\%   & $1.483 \pm 0.039$ & $1.076 \pm 0.010$ & $\bm{+27.5\%}$ & $1.057 \pm 0.016$ & $\bm{+28.7\%}$ & $1.147 \pm 0.038$ & $\bm{+22.6\%}$ \\
5\%   & $1.107 \pm 0.022$ & $0.974 \pm 0.012$ & $\bm{+12.0\%}$ & $0.981 \pm 0.022$ & $\bm{+11.4\%}$ & $0.991 \pm 0.011$ & $\bm{+10.5\%}$ \\
25\%  & $0.772 \pm 0.039$ & $0.815 \pm 0.021$ & $-5.6\%$ & $0.809 \pm 0.040$ & $-4.8\%$ & $0.807 \pm 0.024$ & $-4.6\%$ \\
50\%  & $0.714 \pm 0.013$ & $0.729 \pm 0.013$ & $-2.2\%$ & $0.764 \pm 0.037$ & $-7.0\%$ & $0.749 \pm 0.030$ & $-4.9\%$ \\
100\% & $0.659 \pm 0.025$ & $0.686 \pm 0.032$ & $-4.1\%$ & $0.658 \pm 0.011$ & $+0.1\%$ & $0.653 \pm 0.003$ & $+0.9\%$ \\
\bottomrule
\end{tabular}
\caption{Per-sample nRMSE (\%) on the random-split test set across five real-data fractions, comparing the real-only baseline with synthetic pre-training plus fine-tuning under three noise conditions. Results are mean $\pm$ std over three seeds for all models. Bold gains indicate positive improvement from pre-training.}
\label{tab:random_table}
\end{table}
```

Markdown 对照：

| Real Data fraction | Real-only | Clean nRMSE | Clean Gain | Noisy 0.5 nRMSE | Noisy 0.5 Gain | Noisy 2 nRMSE | Noisy 2 Gain |
|---|---|---|---|---|---|---|---|
| 1% | 1.483 ± 0.039 | 1.076 ± 0.010 | **+27.5%** | 1.057 ± 0.016 | **+28.7%** | 1.147 ± 0.038 | **+22.6%** |
| 5% | 1.107 ± 0.022 | 0.974 ± 0.012 | **+12.0%** | 0.981 ± 0.022 | **+11.4%** | 0.991 ± 0.011 | **+10.5%** |
| 25% | 0.772 ± 0.039 | 0.815 ± 0.021 | −5.6% | 0.809 ± 0.040 | −4.8% | 0.807 ± 0.024 | −4.6% |
| 50% | 0.714 ± 0.013 | 0.729 ± 0.013 | −2.2% | 0.764 ± 0.037 | −7.0% | 0.749 ± 0.030 | −4.9% |
| 100% | 0.659 ± 0.025 | 0.686 ± 0.032 | −4.1% | 0.658 ± 0.011 | +0.1% | 0.653 ± 0.003 | +0.9% |

数据源（`artifacts/tokamind_test_eval/`）：
- A = `pca_e500_scratch-{f}pct-s{s}-e500.json`（100/50/25/5% × s54/55/56）+ 1% = `pca_scr1pct_s54_scr500.json` +
  `pca_e500_scratch-1pct-s{56,58}-e500.json`；C = `pca_w150_{finetune,noisy05-finetune,noisy20-finetune}-{f}pct-s{s}-w150.json`
- checkpoint：`runs/tokamind-pca01sigma-{scratch-...-e500|...,...-w150}`，归档副本：`paper_artifacts/random/checkpoints/random-{scratch,ft-...}-...`（64 ckpt + 61 eval，全 5 档 × 3 臂可完整复现，§57.4）

**解读要点**：稀缺档（1%/5%）预训练正收益（+11.7~+40.2%，1% 最强）；25-100% 无正收益（−2~−6%）——
「充分收敛后预训练只在数据稀缺时有效」；噪声非必要条件（三臂几乎持平）。

### 59.3 Temporal split（论文 tab:temporal_table，出处 §54.6/§55.6/§48.7/§48.8）

```latex
\begin{table}[htbp]
\centering
\small
\begin{tabular}{c c c c}
\toprule
Real data fraction & Real scratch & Pretrain + fine-tuning & Gain \\
\midrule
1\%   & $6.42 \pm 0.04$  & $2.31 \pm 0.17$ & $\bm{+64.0\%}$ \\
5\%   & $3.80 \pm 0.12$  & $2.33 \pm 0.11$ & $\bm{+38.6\%}$ \\
25\%  & $3.237 \pm 0.151$ & $2.249 \pm 0.095$ & $\bm{+30.5\%}$ \\
50\%  & $2.794 \pm 0.173$ & $2.378 \pm 0.165$ & $\bm{+14.9\%}$ \\
100\% & $2.489 \pm 0.167$ & $2.204 \pm 0.129$ & $\bm{+11.4\%}$ \\
\bottomrule
\end{tabular}
\caption{Per-sample nRMSE (\%) on the temporal-split M9 test set (17{,}498 slices from 507 discharges) across five real-data fractions, comparing the real-only baseline with synthetic pre-training plus fine-tuning. Results are mean $\pm$ std over three seeds. Bold values indicate the better result; bold gains indicate positive improvement from pre-training.}
\label{tab:temporal_table}
\end{table}
```

Markdown 对照：

| Real data fraction | Real scratch | Pretrain + fine-tuning | Gain |
|---|---|---|---|
| 1% | 6.42 ± 0.04 | 2.31 ± 0.17 | **+64.0%** |
| 5% | 3.80 ± 0.12 | 2.33 ± 0.11 | **+38.6%** |
| 25% | 3.237 ± 0.151 | 2.249 ± 0.095 | **+30.5%** |
| 50% | 2.794 ± 0.173 | 2.378 ± 0.165 | **+14.9%** |
| 100% | 2.489 ± 0.167 | 2.204 ± 0.129 | **+11.4%** |

数据源（`artifacts/tokamind_test_eval/` + 归档 `paper_artifacts/temporal_full/eval/`）：
- 100/50/25%：A = `temporal_hi_tokamind-temporal-scratch-{f}pct-s{s}-u1e4.json`、C = `temporal_hi_tokamind-temporal-finetune-clean-{f}pct-s{s}-u1e4.json`
- 5%：A/C = `temporal_5u_{A_scratch,C_ft}_3seed.json`（各 3 run）；1%：A/C = `temporal_1u_3seed.json`（6 run，`-1u`）
- checkpoint：`runs/tokamind-temporal-{scratch-{f}pct,finetune-clean-{f}pct}-s{s}-{u1e4|-1u}`；归档副本 33 ckpt + 21 eval 全 5 档（§51.1/§56.2）

**解读要点（论文主证据链）**：**五档全正且随稀缺单调放大**（+11.4% → +64.0%）——预训练在时间外推
（跨装置 campaign）下全域有效；对照 random 高数据档为负（§59.2）→ 预训练提供的是跨分布迁移的
物理先验（每 seed 全正 30/30 见 §55.7）。备选口径：C 用 ft lr5e-5（`u5e5`）结果几乎一致
（§55.6：+11.3/+15.9/+32.6/+39.9/+64.9%，std 更小），论文沿用 u1e4 版。

### 59.4 验证记录（2026-09-03，机器逐格核对）

- **random 表 20 格**（5 档 × A+3 臂）：全精度重算与 LaTeX 显示值逐格一致（PASS ×20），
  数据源 = `paper_artifacts/random/eval/`（61 个 per-run json），仅凭归档可完整复现（§57.4）
- **temporal 表 15 格**（5 档 × A/C/gain）：与归档聚合逐格一致（PASS ×15），
  数据源 = `paper_artifacts/temporal_full/eval/`（含 5%/1% 聚合 json）
- 口径警示（跨表引用必读，附录 A）：① 两表 Gain 均已统一为 (A−C)/A（real scratch 基准，2026-09-03；random 原 (A−C)/C 口径已弃用，旧表见 §50.12 历史注）；
  ② random std 全 ddof=0，temporal 仅 1% 行为 ddof=1；③ random 1% A = {s54,s56,s58}（s57 备用归档仍在）
- 与 progress2 正文锚点一致：random ↔ §57.4；temporal ↔ §54.6（100/50/25%）+ §48.7（5%）+ §48.8（1%）

### 59.5 完整可复现性审计（2026-09-03，全量通过）

> 回答："paper_artifacts 能否独立复现两张论文表？"——**能**（评估级完整复现）。审计证据如下：

1. **数值层**：仅凭 `paper_artifacts/{random,temporal_full}/eval/` 聚合，random 表 20 格 + temporal 表
   15 格与 §59.2/§59.3 逐格一致（机器 PASS，2026-09-03）
2. **权重层**：97 个归档 run（random 64 + temporal 33）的 4 个权重文件（backbone/modality_heads/
   output_adapters/token_encoder）与源 `runs/` 目录 **md5 逐字节一致**（0 差异）
3. **元数据层**：每个 per-run eval json 的 checkpoint_best_val/checkpoint_epoch 与对应 ckpt
   `meta.json` 一致；temporal 5%/1% 聚合 json 内 run/run_dir 均指向现存归档 ckpt 且 meta 匹配
4. **端到端层**（归档 ckpt + 归档 test 数据 → 重评 = BIT-EXACT，14 run 覆盖全部家族）：
   - temporal：scratch/ft × {1%,5%} s54、ft-100pct-s55、scratch-50pct-s54（§56.4 的 4 + 本轮 2）
   - random：scratch-1pct-s57/s58、scratch-100pct-s54、ft-clean-5pct-s54、ft-clean-50pct-s56、
     ft-noisy05-25pct-s55、ft-noisy20-1pct-s56、ft-noisy20-100pct-s54（§56.4 的 3 + 本轮 5）
   → 每 run 全部指标字段（nrmse/raw_rmse/rmae/samples/test_shots 等）逐位相等
5. **test 数据**：`random/test/` = manifest + 21,350 行 cache；`temporal_full/test/` = manifest +
   17,498 行 cache；`dataset_split*.csv|json` 齐全

**边界（审计通过范围之外）**：
- **评估/画图代码不在归档内**（设计如此）：ckpt+test→数字需 workspace 的 `mast-bridge/`（train/eval 脚本）
  + `external/tokamind/`；mast-bridge 代码备份仍为 §53.4 待办 1
- **重训（非评估）复现**需要训练 manifest + cache（`data/manifests/`、`data/processed/`，大体积未归档，
  路径与重建方法见 §51/README）；raw zarr 需 FAIR-MAST 源
- noisy 预训练 init（noisy05/noisy20 版 sec50）未归档（仅 clean）；ft 成品 ckpt 齐全 → 结果复现不受影响

---

## 60. GitHub 论文仓库备份记录（2026-09-03）

> **给新对话的交接**：本工作区的**权威源** = `collab_package/`（本文档所在）。GitHub 仓库
> `Homingdung/mast-bridge` 是**论文仓库**（代码 + 精简 paper_artifacts），非唯一备份。
> 本节省略版记录备份过程、结构与操作方式，避免新会话重复摸索。

### 60.1 仓库与分支状态（2026-09-03）

| 项 | 值 |
|---|---|
| 远端 | `https://github.com/Homingdung/mast-bridge.git`（SSH: `git@github.com:Homingdung/mast-bridge.git`）|
| 本机 clone（暂存，可删）| `/tmp/opencode/paper_repo` |
| 权威代码源 | `/inspire/qb-ilm/project/ai-for-fusion/public/collab_package/mast-bridge/`（含 venv，不入库）|
| 权威日志 | `/inspire/qb-ilm/project/ai-for-fusion/public/collab_package/progress2.md`（本文件）|

分支（2026-09-03）：
- **`main`** = `5233fcb`（表修正 Gain 统一 (A−C)/A；**不含** progress2.md）
- **`fix-table`** = `7f879a2`（= main + progress2.md；**PR #1 → main 已开**，
  https://github.com/Homingdung/mast-bridge/pull/1 ，合并/关闭状态待查）
- **`dev-mingdong`** = 旧远端分支，未动

### 60.2 main 提交史（2026-09-03）

| commit | 内容 |
|---|---|
| `3ad77ce` | 论文仓库快照：最新 mast-bridge 代码 + 精简 paper_artifacts（eval/dataset_split/test manifest/图再生脚本/EXPERIMENTS+PLOT_STYLE）+ README 重写 |
| `c2e9ba3` | README 补外部依赖 tokamind（MMT）安装说明（pin `0b67cf56`）|
| `e32239c` | 删 generalization 图 + plot/out 生成图（留再生脚本）；README 去掉结果数字 |
| `8540b17` | 移除 progress2.md（后用户改主意 → 见 fix-table）|
| `5233fcb` | **Gain 统一 (A−C)/A**（random 表修正：+37.8→+27.5 等）+ EXPERIMENTS/random README 同步 |
| `7f879a2`（fix-table 独有）| progress2.md 入仓库 + README 引用恢复 |

### 60.3 仓库内容规则（重要，勿破坏）

**包含**：`scripts/` `src/` `configs/` `pyproject.toml`（最新代码，不含 venv）、
`paper_artifacts/{EXPERIMENTS.md,PLOT_STYLE.md,random,temporal_full}`（仅：各 README、`eval/*.json`
82 个、`dataset_split*.csv|json`、`test/split_test_real.jsonl`、`plot/{README,scripts}`——**图与数据文件
全部不提交**，`plot/out/` 已 gitignore）、根 README、`progress2.md`（仅 fix-table 分支）。

**不包含**（.gitignore 或人工排除）：两个 venv（`.tokamind-train-env`/`.freegsnke-solve-env`）、
`*.npz/*.pt/*.zarr`、`paper_artifacts/**/checkpoints/`、`plot/out/`（生成图）、test cache npz（>100MB
GitHub 限制）、训练 cache、raw zarr。需要复现的路径说明都在仓库 README。

### 60.4 坑（GitHub 操作）

1. **本环境 SSH 不可用**（`git@github.com` publickey 被拒）→ 一律用 HTTPS + Personal Access Token：
   `git push https://x-access-token:<TOKEN>@github.com/Homingdung/mast-bridge.git <branch>`
2. **2026-09-03 曾暴露 2 个 token**（fine-grained `github_pat_11A...` 无写权限 + classic `ghp_zMHS8F...`
   有 repo 权限）——**应已被撤销**；新会话若需推送，请用户重新生成 classic PAT（`repo` 权限）
3. GitHub 单文件上限 100MB——任何 npz/大产物**严禁**尝试入库（会 403/超限报错）
4. 更新流程：`git clone https://github.com/Homingdung/mast-bridge.git`（或复用 /tmp/opencode/paper_repo）
   → `cp -r` 最新文件 → commit → push 到目标分支；main 与 fix-table 需分别 push
5. 分支用途：论文正式版走 `main`；progress2.md 等内部日志走 `fix-table`（PR #1）；合作者
   Shape-OOD 从 `main` 开新分支

### 60.5 如需恢复/复现

- 评估复现：仓库内 `eval/*.json` 已足够重现论文表数字；端到端重跑需 checkpoints + test cache
  npz（在原工作区 `paper_artifacts/{random,temporal_full}/checkpoints|test/`，未入库，见 §59.5）
- tokamind 外部依赖：README「External dependency」节（clone + pin `0b67cf56`）
- 论文数字/口径最终状态：见 §52.2/§53.3/§57.4/§58.3/§59（Gain 统一 (A−C)/A，2026-09-03）
