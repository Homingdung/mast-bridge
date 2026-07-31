# 合成磁诊断量与训练数据集设计

## 目标

为严格通过 FreeGSNKE `1e-8` 收敛筛选的合成平衡样本生成磁诊断输入，并构建
真实数据、仿真数据、真实加仿真数据三组可直接用于 TokaMind 的 manifest。模型输入
统一为等离子体电流、flux loops、poloidal pickup probes 和 active coil currents，
输出为 `65 x 65` 的 `psi`。

## 数据契约

每个合成样本目录新增 `diagnostics.npz`，包含：

- `schema_version`
- `target_time`
- `magnetics_ip`
- `flux_loop_names`、`flux_loop_values`
- `pickup_names`、`pickup_families`、`pickup_values`
- `active_coil_names`、`active_coil_values`
- `flux_loop_scale`

字符串数组使用 NumPy Unicode dtype，读取时不启用 pickle。所有数值必须 finite，
名称必须唯一，名称数组和数值数组长度必须一致。

## 物理约定

- flux loop 使用 FreeGSNKE 官方 `tokamak.probes.calculate_fluxloop_value(eq)`，
  并乘 `2*pi`，从 `Wb/(2*pi)` 转为 MAST Level 2 的 `Wb`。
- pickup 使用 FreeGSNKE 官方 `tokamak.probes.calculate_pickup_value(eq)`。
- machine pickle 只在临时副本中修正 flux-loop 几何映射以及 OBR/OBV 方向，原始
  下载数据不修改。
- `magnetics_ip` 对仿真样本取扰动后的 Lao85 `Ip`；真实样本继续读取
  `magnetics/ip`。
- active coil currents 对仿真样本取求解时实际使用并写入 `metadata.json` 的值。

## 已有样本补算

已有 `equilibrium.npz` 保存的是总 poloidal flux。补算脚本按以下步骤重建状态：

1. 从 `metadata.json` 恢复 machine、Lao85 参数和实际 coil currents。
2. 创建与保存网格一致的 FreeGSNKE equilibrium。
3. 计算 coil flux，并令 `plasma_psi = saved_total_psi - coil_psi`。
4. 用保存的 Lao85 参数和边界 flux 在最终 `psi` 上重建 `jtor`。
5. 初始化 FreeGSNKE probes，计算并保存合成诊断量。

该过程不执行 Grad-Shafranov 迭代求解，不改变已经通过严格筛选的 `psi` 标签。
脚本必须检查重建后的总 `psi` 与保存值一致。

## 数据集构建

构建 diagnostics 训练 manifest 时，synthetic 样本必须同时满足：

- 已在 `*_synthetic_accepted.jsonl` 中，即收敛且 final tolerance 不高于 `1e-8`；
- `equilibrium.npz` 有效；
- `diagnostics.npz` 存在且通过格式和 finite 检查。

真实样本由通过上述条件的 synthetic 样本对应的唯一
`(parent_shot, target_time)` 构建。三组数据按 parent shot 划分，避免同一炮的真实
时间片和仿真子样本跨越训练集与验证集。

## 错误处理与验证

批处理对单个失败样本写入 report 并继续运行；默认跳过已有有效
`diagnostics.npz`，支持断点续跑。单元测试覆盖 NPZ 往返、无效数据拒绝、synthetic
训练特征读取和 diagnostics manifest 过滤；端到端 smoke test 至少补算一个真实
accepted 样本并确认输出 finite、通道数非零且训练 dry-run 能加载混合数据。
