from __future__ import annotations

from pathlib import Path
from typing import Any

from mast_bridge.mast.machine_config import MachineGeometry


def machine_build_kwargs(machine: MachineGeometry) -> dict[str, Path]:
    """Map normalized machine names to FreeGSNKE's build_machine API."""
    return {
        "active_coils_path": machine.files["active_coils"],
        "passive_coils_path": machine.files["passive_coils"],
        "limiter_path": machine.files["limiter"],
        "wall_path": machine.files["wall"],
        "magnetic_probe_path": machine.files["magnetic_probes"],
    }


def build_machine(machine: MachineGeometry, **kwargs: Any) -> Any:
    """Build a FreeGSNKE machine lazily; solving remains a separate operation."""
    from freegsnke import build_machine as freegsnke_build_machine

    build_kwargs = machine_build_kwargs(machine)
    build_kwargs.update(kwargs)
    return freegsnke_build_machine.tokamak(**build_kwargs)
