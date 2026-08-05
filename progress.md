# mast-bridge 进度记录

## 1. FAIR-MAST Level 2 全量数据下载

- **日期**:2026-08-03 ~ 2026-08-04
- **数据源**:`s3://mast/level2/shots/<shot>.zarr` @ `https://s3.echo.stfc.ac.uk`(匿名,STFC Echo)
- **工具**:s5cmd v2.3.0(8 分片并行,每片 128 workers,断点续传)
- **本地位置**:`../data/raw/mast/`
- **脚本**:`scripts/download_all_mast_shots.py`、`scripts/monitor_download_all.py`

### 下载统计

| 项目 | 数量 |
|---|---|
| 远程 shot 总数 | 11,573 |
| 已下载 shot | 11,573 |
| **5 个必需 group 齐全**(equilibrium/magnetics/pf_active/pf_passive/wall) | **11,374** |
| 缺必需 group(远程本身缺失) | 199 |
| **含 `equilibrium/dpressure_dpsi` + `f_df_dpsi`**(可用于 Lao85 拟合) | **11,306** |
| 有 equilibrium 组但无 EFIT profile 数组 | 68 |

- 存储:约 500 GB 逻辑字节(千万级小文件,磁盘占用预计 0.6–1.0 TB)
- 下载过程 0 个分片失败;首次全量后重跑一次续传,结果不变,确认无遗漏

### 可用数据

| 口径 | 数量 | 占比 |
|---|---|---|
| 远程全部 shot | 11,573 | 100% |
| 已下载 | 11,573 | 100% |
| 5 个必需 group 齐全 | 11,374 | 98.3% |
| **可用于 Lao85 拟合**(含 `dpressure_dpsi` + `f_df_dpsi` 等全部拟合输入) | **11,306** | 97.7% |

- 可用 = 完整组 − 无 profile 数组 = 11,374 − 68 = **11,306 炮**
- 不可用的 267 炮:199 缺必需 group + 68 无 EFIT profile(二者无重叠,均远程源头缺失)

### 清单文件(位于 `../data/raw/mast/`)

- `.all_level2_shots.txt` — 全部 11,573 炮
- `.all_level2_complete.txt` — 11,374 炮(5 group 齐全)
- `.all_level2_incomplete.txt` — 199 炮(缺必需 group)
- `.missing_efit_profile_arrays.txt` — 68 炮(无 EFIT profile 数组,拟合时排除)
- `.download_all.log` / `.download_progress.log` — 下载日志与进度快照

### 说明

- 199 炮缺 group、68 炮缺 profile 数组均已在远程源头确认缺失,非下载问题,无法通过重下解决。
- 下一阶段:对 11,306 炮批量生成 machine pickle 与 Lao fit NPZ。

## 2. Machine Pickles 批量生成

- **日期**:2026-08-04
- **环境**:`.mast-process-env`(原 macOS 拷贝 venv 在本机不可用,已按 README 2.2 用 Python 3.12.3 重建:numpy 2.5.1 / zarr 3.2.1 / xarray 2026.7.0 / scipy 1.18.0 / matplotlib 3.11.1)
- **脚本**:`scripts/run_machine_batch.py`(32 并行、断点续跑、按 shot 记录 ok/failed)
- **输出**:`../data/raw/mast/machine/<shot>/` 下 5 个 pickle(active_coils / limiter / magentic_probes / passive_coils / wall)

### Machine 统计

| 项目 | 数量 | 占比 |
|---|---|---|
| 尝试生成 | 11,306 | 100% |
| **5 个 pickle 齐全** | **11,148** | 98.6% |
| 失败(缺 `MAST_active_coils.pickle`) | 158 | 1.4% |

- 158 个失败原因:该炮 `pf_active` 组缺少几何信号(如 `current_channel`、`p2_inner_lower_height`、`sol_width`、`sol_r`、`p3_lower_height` 等),为数据本身缺失,无法建模线圈
- 代码修复 1 处:`ACTIVE_GROUPS` 中 P6L/P6U 由"必需"改为"有则建模"(FAIR-MAST 多数炮无 P6 线圈,只有 13 个通道的老炮有);修复后 6,338 炮从失败转为成功,13 炮复现集的 13 通道行为不变,单元测试通过
- 注:批量脚本曾与首轮任务并发写同一 report,最终报告已按文件系统重建

### 清单文件(位于 `../data/raw/mast/`)

- `.machine_complete.txt` — 11,148 炮(machine 齐全)
- `.machine_failed.txt` — 158 炮(生成失败)
- `.machine_batch_report.jsonl` — 每炮 ok/failed 明细(已重建为单轮干净记录)

## 3. 后续可用数据(全流程)

递进筛选口径:

| 阶段 | 数量 | 筛掉原因 |
|---|---|---|
| 远程全部 | 11,573 | — |
| 下载 + 5 group 齐全 | 11,374 | 199:远程缺必需 group |
| + 有 EFIT profile 数组(拟合输入) | 11,306 | 68:无 `dpressure_dpsi` 等 |
| + machine pickle 生成成功 | **11,148** | 158:`pf_active` 缺几何信号 |

- **最终可用于后续全流程 = 11,148 炮**(machine ⊆ 拟合可用 ⊆ 完整组,逐层无重叠)
- 同时满足:5 必需 group、Lao85 拟合输入、machine pickle 齐全 → 可走通「Lao 拟合 → FreeGSNKE 求解 → synthetic → 训练」
- 清单:`../data/raw/mast/.pipeline_usable.txt`(11,148 炮)
- 后续 `build_lao_fit_npz.py` 直接以 `.pipeline_usable.txt` 为 shot 列表

## 4. Lao85 拟合(已完成)

- **日期**:2026-08-04
- **输入**:11,148 炮(`.pipeline_usable.txt`)
- **输出**:`../data/processed/real/lao_parameter_ensemble/all_zarr_lao_parameter_fits.npz`
- **脚本**:`scripts/run_lao_fit_batch.py`(16 并行、容错、断点续跑)

### 拟合结果

| 项目 | 数量 |
|---|---|
| 拟合 shot | 11,148 / 11,148(0 失败) |
| 拟合行数 | **818,139** |
| 每炮行数 | 中位 73(2–136) |
| alpha/beta | 全部有限 |

### 时间片统计(筛选口径:11,148 炮)

| 口径 | 数量 | 说明 |
|---|---|---|
| 原始 equilibrium 时间片 | **1,316,020** | 11,148 炮的原始 EFIT 时间轴(每炮中位 116,53–618) |
| 拟合成功行(全时域) | 818,139 | 需 profile 有限,~62% 保留 |
| [0.12,0.24] 窗内时间片 | 266,256 | 11,148 炮均有 ≥1 片(每炮平均 ~24) |
| [0.12,0.24] 窗内拟合行 | 256,425 | 窗内拟合筛掉 ~4% |

### Pilot 基准(求解前摸底,2026-08-04)

- 120 点分层抽样(campaign × 20),部分完成即停;干净统计口径(剔除首启脏数据):
  - 典型单解耗时 1.1–4.5 分钟(均值 ~2.7,65×65、tol 1e-8、max 500 iter)
  - 合法收敛样本 ≤253 次迭代、≤5 分钟;病态样本 500 次迭代跑满(50+ 分钟)后仍被拒 → 需 5 分钟墙钟超时兜底
  - 快速数据崩溃 ~14%:`pf_passive` 电流在目标时刻 NaN(**可预检 100% 命中、0 误报**)
  - 已跑完样本严格接受率 ~95%;总接受率 ~82%(无预检)/ ~90%(有预检)
- 本机 128 核(双路 Xeon Gold 6530),求解可 64–96 路并行,线性扩展
- 方案时间估计(预检 + 5 分钟超时 + 64 并行,~3.1 分钟/解):A 11k 解 ~9 h;B 22k 解 ~18 h;C 66k 解 ~54 h;D 88k 解 ~71 h

### 破裂排除与平顶平台筛选(2026-08-04)

**方法**:候选 (shot, time) 需同时满足两条客观判据,不做主观/注释判断:

1. **平顶窗**:`t ∈ [plasma_flat_top_start_time, plasma_flat_top_end_time]`(来自 FAIR-MAST shot 元数据,11,148 炮全部有值)
2. **平台电流**:`Ip(t) ≥ 0.85 × 炮内 Ip 峰值`(Ip 取 magnetics/ip 在拟合时刻的插值,即 fit NPZ 的 ip 列)

**筛选结果**([0.12,0.24] 窗内 256,425 个拟合行):

| 条件 | 行数 | 占比 |
|---|---|---|
| 原始窗内拟合行 | 256,425 | 100% |
| + 在元数据平顶窗内 | 192,843 | 75.2% |
| + Ip ≥ 85% 炮内峰值 | 223,602 | 87.2% |
| **两者都满足(干净平顶)** | **187,703** | **73.2%** |

- 覆盖 **10,415 炮**(占可用 11,148 的 93.4%),每炮干净切片中位 21 片(1–24)
- **破裂排除靠 Ip 判据而非注释**:注释含 "disrupt" 的 862 炮与 Ip 崩溃(Ip 跌到峰值 50% 以下,420 炮)几乎不重合(仅 33 炮),注释标签不可靠;`Ip ≥ 85% 峰值` 天然排除破裂段/启动段/下降段
- 干净平顶池上的方案规模(×1 变体):K=1 → 10,415 解;K=2 → 20,766 解;K=3 → 30,897 解;K=4 → 40,608 解
- **决定(2026-08-04)**:采用切片级筛选,不做整炮剔除。破裂炮破裂前的平顶片是真实存在的有效平衡态,保留重建;`Ip ≥ 85% 峰值` 只排除启动/下降/破裂塌缩段本身。

**[0.12, 0.24] 窗口依据**(10,771 炮验证,Ip ≥ 85% 平台判据):

| 指标 | p10 | p25 | p50 | p75 | p90 |
|---|---|---|---|---|---|
| 平顶开始(Ip 达 85% 平台) | 0.075 | 0.095 | 0.105 | 0.135 | 0.17 |
| 平顶结束(Ip 仍 ≥85% 平台) | 0.235 | 0.30 | 0.34 | 0.40 | 0.475 |

- 下界 0.12:0.12s 时 68.6% 炮已在平台上;更早会混入启动段(limiter-limited 等离子体,EFIT 质量差)
- 上界 0.24:0.24s 时 89.9% 炮仍在平台上;更晚会让平顶较短的老炮掉出(M5 平顶结束中位 0.33s vs M9 0.395s)
- 完整覆盖 [0.12,0.24] 的炮占 61%;窗口为 5 个 campaign 的公共平顶交集,配合每炮元数据平顶窗 + Ip 判据三级筛选

### 论文统计图数据源(可复现)

- **脚本**:`scripts/build_clean_pool_analysis.py`(唯一数据源,重跑即可再生所有筛选统计)
- **输出目录**:`../data/processed/real/clean_pool/`
  - `clean_pool.csv` — 全部 818,139 拟合行 + 每行筛选标记(in_window / in_flat_top / ip_ok / clean)
  - `per_shot_flat_top_stats.csv` — 每炮 plateau、平顶 onset/end、clean 片数
  - `funnel_counts.json` — 漏斗数字 + 平顶起止百分位(含窗口条件化口径,与上表一致)
- 可画图项:筛选漏斗(11573→11374→11306→11148→818,139→187,703)、平顶起止时间分布、Ip 阈值示例曲线(干净炮 vs 破裂炮)、每炮时间片数分布

### Lao85 拟合精度验证(2026-08-04,求解前最后确认)

- **抽样**:60 炮 × 4,481 行,重建 p′(ψ)/ff′(ψ) 并与 EFIT 原始 profile 对比
- **结果**:n=2(4 参数)起重建最大相对误差即 0.00%,100% 行精确还原;当前 n=3(6 参数)完全足够,甚至冗余
- **原因**:FAIR-MAST 存储的 `dpressure_dpsi`/`f_df_dpsi` 本身就是低阶多项式(MAST EFIT 的 Lao 形式),多项式基底重建为精确还原
- **语义确认**:拟合设计矩阵(ψⁱ−ψ³,保证 p′(1)=0)与 FreeGS4E `Lao85(alpha_logic=True)` 约定逐项一致,alpha/beta 直接可用;`Ip_logic=True` 额外把 Jtor 归一化到精确 Ip

## 5. 生产求解(方案 A,已完成)

- **日期**:2026-08-04 启动,约 4 小时完成(64 路并行)
- **采样**:干净池(去元数据窗版,Ip 判据 `[0.12,0.24] ∩ Ip≥85%`)= 223,602 片 / 10,599 炮;每炮取中位时间片 1 个 × 1 变体
  - 注:元数据平顶窗被验证为过度筛选(184 炮被完全误杀,如 11780 实际全程 92–96% 平台),生产改用 Ip 判据
- **预检**:passive 电流 NaN 过滤,10,599 → 10,000 行(599 点剔除)
- **13 炮覆盖**:全部 13 炮(含 11780)在列;旧 13 炮 synthetic(624 样本)与旧 manifest(10 个)已归档至 `../data/processed/archive/`
- **求解配置**:64 路并行,65×65、tol 1e-8、max-iterations 500、每样本超时 300s
- **关键修复(线程过订阅)**:首启 64 进程 × OpenBLAS 默认 128 线程 = 8,255 线程,首波 92% 超时;限制 `OMP/OPENBLAS/MKL/NUMEXPR_NUM_THREADS=1` 后每进程 ~5.6 线程,首波 207/213 成功(97%),单解 ~3 分钟
- **产物**:`../data/processed/synthetic_production/<shot>_t<time>_v000/`

### 生产求解结果

| 阶段 | 数量 |
|---|---|
| 变体行(预检后) | 10,000 |
| **严格过滤 accepted(≤1e-8)** | **7,455(74.6%)** |
| rejected | 2,420 |
| 进程失败(无产物) | 125 |

**rejected 原因明细**(2,420):

| 原因 | 数量 | 说明 |
|---|---|---|
| `solver_not_converged` | 2,395 | 500 次迭代内未达 1e-8;残差中位 4.8e-3,p90 4.8e-2,远高于阈值,无边缘争议样本 |
| `metadata_missing` | 25 | 求解进程异常退出无元数据 |

**accepted 明细**(7,455):

- 覆盖炮数 / 时间片分布:1 变体/炮 × 1 时间点(每炮取干净片的中位时刻),见 manifest 可按 `parent_shot`/`target_time` 分组
- 13 炮:5 炮 accepted(11766/11768/11769/11771/11776,均 t≈0.18),8 炮 rejected(未收敛)

- 核对(全链路一致):10,000 = 7,455 accepted + 2,420 rejected + 125 failed;accepted/rejected 与样本目录一一对应、无重叠、无目录超出 CSV
- 13 炮全部覆盖:5 炮 accepted(11766/11768/11769/11771/11776),8 炮 rejected(未在 500 次迭代内收敛)
- 排障记录:分片 18 因"半成品目录(metadata 缺 variant_row)"触发批处理崩溃,152 行漏跑;已补丁 `run_lao85_variant_solve_batch.py`——现有输出校验失败时改为清除并重算,不再崩溃(单元测试通过)
- 接受率 74.6% 低于 Pilot 估计(~90%):Pilot 仅 33 个完整样本,统计量小
- manifest:`data/manifests/production_a_synthetic_accepted.jsonl`(7,455)/ `production_a_synthetic_rejected.jsonl`(2,420)

## 6. 合成磁诊断 + 三组 Manifest(已完成)

- **日期**:2026-08-04
- **脚本**:`scripts/build_synthetic_magnetic_diagnostics.py`(32 分片并行,7,452/7,455 生成,0 失败)

### 诊断输入维度(94 维公共特征,训练固定 schema)

| 类型 | 维度 | 通道 |
|---|---|---|
| `target_time` | 1 | 目标时刻 |
| `magnetics_ip` | 1 | 等离子体电流 |
| `flux_loop` | 10 | CC03/CC05/P3U(1,4)/P4L(1,4)/P4U(4)/P5L(1,4)/P5U(1) |
| `pickup` | 69 | CCBV 39 + OBR 15 + OBV 15 |
| `coil_active` | 13 | SOL + P2IL..P6U(12 PF) |
| **合计** | **94** | 输出:65×65 `equilibrium/psi` |

- synthetic 完整字段 137(44 磁通环 + 78 探针 + 13 线圈 + ip + time),与真实数据公共有限通道 = 94
- 噪声标定(40 炮真实数据平顶段高频残差 σ):flux_loop 7.5e-3–4.6e-2 Wb;pickup CCBV 5.6e-3–1.8e-2 T、OBR/OBV ~2e-3 T;coil 4–260 A;Ip 用保守值 ~1 kA(实测残差含低频,偏大)
- **噪声能力已实现**:`--noise-profile configs/diagnostic_features/noise_profile.json` + `--noise-seed`(逐通道独立高斯,每样本种子由 sample_id 派生,可复现);已生成 `diagnostics_noisy.npz`(7,455),加噪/无噪对比实验留待后续

### 加噪样本生成方法

**噪声标定(实测法,2026-08-04)**:
1. 随机抽样 40 炮真实数据(可用集),对 94 维 schema 中每个通道取平顶段 [0.12,0.24] 的时间序列
2. 用 21 点滑动平均分离低频趋势,残差(高频成分)的标准差作为该通道的噪声 σ
3. 结果:flux_loop σ≈7.5e-3–4.6e-2 Wb;pickup CCBV σ≈5.6e-3–1.8e-2 T、OBR/OBV σ≈2e-3 T;coil_active σ≈4–260 A;magnetics_ip 实测残差含低频成分偏大(36 kA),采用保守值 ~1 kA(约平台 Ip 的 0.2%)
4. 93 个可加噪特征全覆盖(94 − target_time),缺测通道用同族中位数补齐

**加噪方式**:
- 独立高斯噪声:每个通道 `value + N(0, σ_channel)`,σ 来自噪声 profile(物理单位,与 diagnostics.npz 单位一致)
- 逐样本可复现:每样本 `seed = (noise_seed + crc32(sample_id)) & 0xFFFFFFFF`,与处理顺序无关
- 不动的量:`target_time`(时间标签)、`flux_loop_scale`(常数换算系数)

**产物**:
- `diagnostics.npz`(无噪,7,455 份)与 `diagnostics_noisy.npz`(加噪,7,455 份)并存于每个样本目录
- 噪声 profile:`configs/diagnostic_features/noise_profile.json`(93 通道 σ)
- 复现命令:在 `build_synthetic_magnetic_diagnostics.py` 后加 `--noise-profile configs/diagnostic_features/noise_profile.json --noise-seed 20260804 --output-name diagnostics_noisy.npz`
- 注意:加噪/无噪对比实验未运行,留待训练阶段(real / clean / noisy / mixed 对比)
- manifest(`--require-synthetic-diagnostics`):

| manifest | 数量 |
|---|---|
| `production_a_diagnostics_real_only.jsonl` | 7,455 |
| `production_a_diagnostics_synthetic_only.jsonl` | 7,455 |
| `production_a_diagnostics_real_plus_synthetic.jsonl` | 14,910 |

### 训练输入 schema 修正(2026-08-04,重要)

- 全量检查(7,457 real 行)发现原 94 维 schema 不成立:**6,564 行(88%)有问题**
  - `coil_active_P6L/P6U` 缺失 4,733 行(63.5% 的炮无 P6 线圈;schema 源自 13 炮实验,那 13 炮恰好都有 P6)
  - 23 个 pickup 通道存在 NaN(CCBV10/22/13/35/06/01/23/02/21/32、OBR19/03/07、OBV07/02/12/19/08/13/14 等,每通道影响 1,200–2,800 行)
- **修正方案**:
  - 缺失 `coil_active_*` 通道按物理含义**补 0**(该炮无此线圈 = 电流 0),代码补丁 `diagnostic_feature_vector`(单测通过)
  - 剔除 23 个有 NaN 的 pickup 通道
  - 新 schema:`configs/diagnostic_features/mast_level2_common_71.json`(71 维 = 1 target + 1 ip + 10 flux + 46 pickup + 13 coil;SHA 校验通过)
- 验证:7,457 real + 7,157 synthetic 行在 71 维下全部有限,0 失败
- 检查脚本:`scripts/check_feature_availability.py`(进度条 + 增量保存 + 断点续跑)

## 7. TokaMind 训练(准备中)

### GPU 训练环境与提速做法(2026-08-04)

1. **GPU 设备**:NVIDIA RTX 4090(51 GB 显存),`NVIDIA_VISIBLE_DEVICES` 挂载后 `torch.cuda.is_available()=True`
2. **使用容器预装 PyTorch**:容器是 NGC PyTorch 镜像(自带 torch 2.8.0a0+cu129,按 numpy 1.x 编译)。训练 venv 用 `python3.12 -m venv --system-site-packages .tokamind-train-env` 继承系统 torch;**不装 numpy 2.x**(破坏 torch 桥接),保留系统 numpy 1.26.4;仅补 zarr 3.2.1/pyyaml/tqdm/psutil + `pip install -e .`、`-e ../external/tokamind`
   - 注意:本机 `/etc/pip/constraint.txt` 把 torch 钉在 2.8.0a0(容器约束),如独立安装 torch 需 `PIP_CONSTRAINT=/dev/null`
3. **训练脚本上 GPU**:`train_tokamind_manifest.py` 增加——模型构建后 `if torch.cuda.is_available(): model = model.cuda()`(训练循环按模型设备搬运 batch)
4. **数据集提速(关键,2 处)**:
   - `ManifestWindowDataset.__getitem__` 原实现**每个 batch 重新读 zarr**(实测 1.6 s/batch),改为 `from_rows` 一次性并行提取后存内存矩阵(`features`/`targets`),`__getitem__` 直接索引 → **0.4 ms/batch**
   - 新增 `--dataset-cache`:预提取的 `(features, psi)` npz(见 `data/processed/training_cache/`),构建从 ~40 分钟/run 降至秒级;缓存按 sample_id 顺序校验
5. **实测性能**:全量 7,157 行 + 缓存构建 ~2 s;GPU 单步 **~14 ms**(89,400 步 ≈ 25 min/run)
6. **实验规模**:单源 100 epochs / mixed 50 epochs(≈89,400 步对齐),batch=8,微调 100 epochs;9 组实验合计 ~4 h(单 GPU 串行)
7. **特征 schema 更新**:大规模数据公共有限特征 = **69 维**(94 维里 P6L/P6U + 23 个 pickup 通道在多数炮缺测),见 `configs/diagnostic_features/mast_level2_common_69.json`(强制纳入版本控制)

## 8. 后续阶段(待更新)

- [ ] 9 组训练(real / synthetic-clean / mixed-clean / 预训练微调 ×2 + noisy 分支 ×4)
- [ ] 300 炮测试集统一评估 + loss 曲线
