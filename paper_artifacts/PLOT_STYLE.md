# 论文散点图风格规范（LOCKED，2026-09-03 定稿）

> 适用：**psi 散点（逐样本均值 / 网格点采样）**与 **j_tor–Ip 散点** 的全部 2×2 combined 图。
> 基准实现（新） = `random/plot/scripts/plot_psi_grid_points.py`（psi 网格点版）；
> 后续散点图一律照此实现，改动前先改本规范。

## 1. 布局与面板语义

- **2×2 网格**：行 0 = A real scratch；行 1 = C pretrain+ft；列 0 = 5%；列 1 = 1%。
- 面板左上文本：`$R = {corr:.4f}$`（坐标 (0.04,0.97)，fontsize 11，ha left/va top）
- 臂标签（灰 0.25）：(0.04,0.88)，fontsize 9 —— psi/jtor 图统一用 `real scratch 5%`、
  `real scratch 1%`、`pretrain + ft 5%`、`pretrain + ft 1%`
- **无 title / suptitle**；**无网格线**（⚠️ 禁止用 `common.style_ax()`——它自带 `ax.grid(alpha=0.18)`；
  只手动去掉 top/right spine）；去掉 top/right spine（只留 x/y）
- figsize = (11.0, 8.6)，dpi=300，`savefig(bbox_inches="tight")`

## 2. 散点/轴约定

- hexbin：`cmap="viridis", mincnt=1`；gridsize = **200**（网格点采样 ~18 万点）或 **80**
  （逐样本均值 ~2.1 万点）
- y=x 参考线：红虚线 `"r--"` lw 1.6 alpha 0.8，取面板数据 lims，`set_aspect("equal")`
- 轴标签只在边界：底行 xlabel、左列 ylabel；tick labelsize=8
- **psi 图**：x = **`EFIT psi / ground truth`**，y = `psi pred`（不写单位——psi 已 raw Wb，
  图例文字足够；禁止再用 `psi true (Wb)` 等写法）
- **j_tor–Ip 图**：x = `$I_p^{\mathrm{meas}}$ (MA)`，y = `$I_p^{\mathrm{pred}}$ (MA)`
- colorbar：`fig.colorbar(hb, ax=ax, shrink=0.72, pad=0.02, aspect=18)`，
  label `count` fontsize 8、tick 7、`outline.set_visible(False)`

## 3. 数据口径（禁止混用）

| 图 | 数据 | 说明 |
|---|---|---|
| psi 网格点采样 | pred_psi/true_psi 逐像素均匀采样 FRAC=0.002、rng=2026 | ~180k 点/面板，corr 在采样像素上算（progress2 §13 图 2）|
| psi 逐样本均值 | pred/true 的 axis=(1,2) 均值 | ~21k 点/面板，corr 在均值上算 |
| j_tor–Ip | lapstar(pred_psi) → J_φ → 面内积分 vs 罗氏线圈 Ip | EFIT j_phi 阈值掩码（progress2 §17）|

## 4. 文件清单（本规范实现者）

| 图（png） | 脚本 | 位置 |
|---|---|---|
| psi_scatter_points_combined_1_5pct.png（网格采样）| plot_psi_grid_points.py | `paper_artifacts/random/plot/scripts/`（自包含，preds 已备）+ `plots/script/` 无旧版 |
| psi_scatter_combined_1_5pct.png（逐样本均值）| plot_psi_scatter_combined.py | 上游 `plots/script/`；归档版 = `paper_artifacts/random/plot/scripts/plot_psi_scatter_combined.py`（自包含）|
| jtor_ip_combined_1_5pct.png | 画图：plot_jtor_ip_combined.py（自包含，读 data/jtor_ip_pairs.npz）；数组重算：compute_jtor_pairs.py（需全量 shot zarr，仅 workspace）| 归档 = `paper_artifacts/random/plot/scripts/`；上游旧版 = `plots/script/plot_jtor_combined.py` |
| 单模型 psi 散点 | plot_psi_scatter.py | `plots/script/`（同规范：mean gs80 / points gs200）|

> 校验数字（2026-09-03 实测，random test 21,350，面板顺序 = scratch5%/scratch1%/ft5%/ft1%）：
> 网格点 R = 0.9982 / 0.9967 / 0.9985 / 0.9981；
> 逐样本均值 R = 0.9963 / 0.9943 / 0.9966 / 0.9951；
> j_tor–Ip R = 0.9636 / 0.9359 / 0.9923 / 0.9933。
