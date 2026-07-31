from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class WorkspaceBootstrapTests(unittest.TestCase):
    def test_full_environment_rejects_python_314(self) -> None:
        from mast_bridge.workspace import full_environment_python_error

        self.assertIsNone(full_environment_python_error((3, 13, 0)))
        self.assertIsNotNone(full_environment_python_error((3, 14, 0)))

    def test_discovers_preferred_external_layout(self) -> None:
        from mast_bridge.workspace import discover_workspace

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "fusion-workspace"
            mast_bridge_root = workspace / "mast-bridge"
            external = workspace / "external"
            for name in ("tokamind", "freegsnke"):
                (external / name).mkdir(parents=True)
            mast_bridge_root.mkdir()

            layout = discover_workspace(mast_bridge_root=mast_bridge_root)

            self.assertEqual(layout.workspace_root, workspace.resolve())
            self.assertEqual(layout.external_root, external.resolve())
            self.assertEqual(layout.tokamind_root, (external / "tokamind").resolve())
            self.assertEqual(layout.freegsnke_root, (external / "freegsnke").resolve())

    def test_discovers_current_sibling_layout(self) -> None:
        from mast_bridge.workspace import discover_workspace

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "fusion-workspace"
            mast_bridge_root = workspace / "mast-bridge"
            for name in ("tokamind", "freegsnke"):
                (workspace / name).mkdir(parents=True)
            mast_bridge_root.mkdir()

            layout = discover_workspace(mast_bridge_root=mast_bridge_root)

            self.assertEqual(layout.tokamind_root, (workspace / "tokamind").resolve())
            self.assertEqual(layout.freegsnke_root, (workspace / "freegsnke").resolve())

    def test_doctor_accepts_current_sibling_layout_without_external_root(self) -> None:
        from mast_bridge.workspace import discover_workspace, run_doctor

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "fusion-workspace"
            mast_bridge_root = workspace / "mast-bridge"
            for name in ("tokamind", "freegsnke"):
                (workspace / name).mkdir(parents=True)
            mast_bridge_root.mkdir()

            layout = discover_workspace(mast_bridge_root=mast_bridge_root)
            report = run_doctor(layout, check_imports=False)

            self.assertTrue(report.ok)
            self.assertNotIn("external_root", report.missing_paths)
            self.assertNotIn("runs_root", report.missing_paths)
            self.assertNotIn("artifacts_root", report.missing_paths)

    def test_writes_local_paths_yaml(self) -> None:
        from mast_bridge.workspace import discover_workspace, write_paths_yaml

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "fusion-workspace"
            mast_bridge_root = workspace / "mast-bridge"
            external = workspace / "external"
            for name in ("tokamind", "freegsnke"):
                (external / name).mkdir(parents=True)
            mast_bridge_root.mkdir()

            layout = discover_workspace(mast_bridge_root=mast_bridge_root)
            output = mast_bridge_root / "configs" / "paths.local.yaml"
            write_paths_yaml(layout, output)

            text = output.read_text(encoding="utf-8")
            self.assertIn(f"workspace_root: {workspace.resolve()}", text)
            self.assertIn(f"tokamind_root: {(external / 'tokamind').resolve()}", text)
            self.assertIn(f"freegsnke_root: {(external / 'freegsnke').resolve()}", text)
            self.assertNotIn("large_model_fusion_root", text)
            self.assertNotIn("mast_data_dir", text)
            self.assertNotIn("tokamark_root", text)

    def test_doctor_reports_missing_external_repo(self) -> None:
        from mast_bridge.workspace import discover_workspace, run_doctor

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "fusion-workspace"
            mast_bridge_root = workspace / "mast-bridge"
            (workspace / "external" / "freegsnke").mkdir(parents=True)
            mast_bridge_root.mkdir()

            layout = discover_workspace(mast_bridge_root=mast_bridge_root)
            report = run_doctor(layout, check_imports=False)

            self.assertFalse(report.ok)
            self.assertIn("tokamind_root", report.missing_paths)
            self.assertIn("freegsnke_root", report.present_paths)
            self.assertNotIn("large_model_fusion_root", report.missing_paths)

    def test_bootstrap_dry_run_does_not_create_workspace_directories(self) -> None:
        from mast_bridge.workspace import bootstrap_main

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "fusion-workspace"
            mast_bridge_root = workspace / "mast-bridge"
            mast_bridge_root.mkdir(parents=True)

            code = bootstrap_main(["--mast-bridge-root", str(mast_bridge_root), "--write-config", "--dry-run"])

            self.assertEqual(code, 1)
            self.assertFalse((workspace / "external").exists())
            self.assertFalse((workspace / "data").exists())
            self.assertFalse((workspace / "runs").exists())
            self.assertFalse((workspace / "artifacts").exists())

    def test_next_steps_include_clone_and_editable_install_commands(self) -> None:
        from mast_bridge.workspace import discover_workspace, format_next_steps, run_doctor

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "fusion-workspace"
            mast_bridge_root = workspace / "mast-bridge"
            mast_bridge_root.mkdir(parents=True)

            layout = discover_workspace(mast_bridge_root=mast_bridge_root)
            report = run_doctor(layout, check_imports=True)
            text = format_next_steps(layout, report, python=Path("/venv/bin/python"), with_deps=False)

            self.assertIn("git clone https://github.com/UKAEA-IBM-STFC-Fusion-FMs/tokamind.git external/tokamind", text)
            self.assertIn("git clone https://github.com/FusionComputingLab/freegsnke.git external/freegsnke", text)
            self.assertNotIn("LARGE_MODEL_FUSION", text)
            self.assertNotIn("external/tokamark", text)
            self.assertIn("/venv/bin/python -m pip install -e", text)
            self.assertIn("--no-deps", text)

    def test_full_install_enables_freegs4e_extra_for_freegsnke(self) -> None:
        from mast_bridge.workspace import discover_workspace, editable_install_commands

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "fusion-workspace"
            mast_bridge_root = workspace / "mast-bridge"
            mast_bridge_root.mkdir(parents=True)
            layout = discover_workspace(mast_bridge_root=mast_bridge_root)

            commands = editable_install_commands(layout, python=Path("/venv/bin/python"), with_deps=True)

        freegsnke_command = next(command for command in commands if "freegsnke" in str(command[-1]))
        self.assertTrue(str(freegsnke_command[-1]).endswith("freegsnke[freegs4e]"))


if __name__ == "__main__":
    unittest.main()
