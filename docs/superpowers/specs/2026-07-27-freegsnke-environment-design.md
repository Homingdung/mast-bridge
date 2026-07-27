# FreeGSNKE Environment Configuration Design

## Goal

让读者激活 `.mast-bridge-env` 后，按仓库提供的一条安装流程即可使用当前 workspace 中的 `external/freegsnke`，并能通过 doctor 验证导入状态。

## Scope

- 保留 `external/freegsnke` 作为外部源码仓库，不复制或修改其源码。
- 使用当前虚拟环境的 Python 执行 `pip install -e external/freegsnke`，并安装 FreeGSNKE 声明的运行时依赖。
- 继续支持只读取 MAST 数据的轻量路径，不强制安装 FreeGSNKE。
- 更新 bootstrap/doctor 的提示和 README，使读者能复现配置。

## Design

`mast_bridge.workspace.editable_install_commands()` 继续生成四个外部仓库的 editable 安装命令；完整环境通过现有 `--with-deps` 让 pip 安装 FreeGSNKE 的 `requirements.txt`。不在 `mast-bridge` 的基础依赖中直接锁定 FreeGSNKE，因为其源码位置是 workspace 级路径，且轻量 MAST 数据路径不需要它。

`bootstrap_workspace.py --install-editable --with-deps` 使用当前解释器安装 `mast-bridge` 和外部仓库。`doctor.py` 保持检查 `freegsnke` 模块，README 明确激活环境、安装外部仓库和验证命令。测试覆盖命令生成以及导入检查提示，避免读者激活了错误的 Python 环境却误以为安装成功。

## Success Criteria

1. `python scripts/bootstrap_workspace.py --install-editable --with-deps` 的安装命令包含 FreeGSNKE editable 安装，并使用当前 Python。
2. 在安装完成的环境中执行 `python -c "import freegsnke"` 成功。
3. `python scripts/doctor.py` 报告 `freegsnke: OK`（其他缺失 workspace 数据仍可单独报告）。
4. 仓库测试通过，且 README 给出 macOS/Linux 和 Windows 的激活后安装路径。

## Alternatives Rejected

- 只在 README 中手写 pip 命令：容易与 bootstrap/doctor 的实际行为漂移。
- 把 FreeGSNKE 复制进 mast-bridge：破坏外部依赖边界，并增加同步维护成本。
