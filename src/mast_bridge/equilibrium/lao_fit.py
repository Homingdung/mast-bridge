from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class LaoProfile:
    """Solver-neutral Lao profile coefficients or coefficient ranges.

    The class deliberately stores the user configuration only. Numerical fitting
    and sampling are separate operations so fitted real-shot results remain
    immutable inputs for later experiments.
    """

    model: str
    pprime_coefficients: list[Any]
    ffprime_coefficients: list[Any]
    sampling: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LaoProfile":
        model = str(data.get("model", "lao85"))
        if not data.get("pprime_coefficients"):
            raise ValueError("pprime_coefficients must not be empty")
        if not data.get("ffprime_coefficients"):
            raise ValueError("ffprime_coefficients must not be empty")
        return cls(
            model=model,
            pprime_coefficients=list(data["pprime_coefficients"]),
            ffprime_coefficients=list(data["ffprime_coefficients"]),
            sampling=dict(data.get("sampling", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "pprime_coefficients": list(self.pprime_coefficients),
            "ffprime_coefficients": list(self.ffprime_coefficients),
            "sampling": dict(self.sampling),
        }
