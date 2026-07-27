# Machine Configuration From Shot Zarr

## Goal

为每个已下载的 MAST shot 从其自身的 Zarr geometry 字段生成 FreeGSNKE 兼容的五个 machine pickle，并保存到 workspace 的 `data/raw/mast/machine/`。

## Design

新增纯 Python 转换模块和 CLI 脚本。转换模块只依赖 Zarr、NumPy 和标准库，不读取 `external/freegsnke/machine_configs/` 中的任何预生成配置。

几何来源固定为：

- `pf_active`：active coils；按 `current_channel` 映射到 `Solenoid`, `P2IL`, …, `P6U`，几何属性来自各 geometry group 的 `r/z/width/height`。
- `pf_passive`：passive coils；每个 geometry group 的几何数组展开为 FreeGSNKE passive-entry 列表。
- `wall`：`limiter_r` 与 `limiter_z` 同时生成 limiter 和 wall 列表。
- `magnetics`：flux-loop 与 pickup probe 几何生成 FreeGSNKE probe 字典。

CLI 默认使用项目 workspace discovery 找到 `data/raw/mast/<shot>.zarr` 和 `data/raw/mast/machine/`，支持显式 `--data-dir`、`--output-dir`，并在目标文件已存在时要求 `--overwrite`。

## Validation

测试使用最小 Zarr fixture，验证五个 pickle 的文件名、关键字段、数组长度和 wall/limiter 内容；CLI 测试验证默认输出和缺失 shot 的错误。真实 shot 上运行生成脚本、`inspect_shot.py`，并在可用时调用 FreeGSNKE machine builder 做兼容性 smoke test。
