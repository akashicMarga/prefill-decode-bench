"""
Shared dataclasses. Both backends produce these — plot_results.py
and the summary printer consume them without knowing which backend ran.
"""

from dataclasses import dataclass, asdict
import json
from pathlib import Path


@dataclass
class SystemInfo:
    backend: str        # "mlx" | "cuda"
    chip: str           # "Apple M3 Max" | "NVIDIA RTX 4090"
    memory_gb: float    # unified memory (Apple) or VRAM (CUDA)
    memory_type: str    # "unified" | "vram"
    model: str


@dataclass
class PrefillResult:
    prompt_tokens: int
    actual_tokens: int
    time_seconds: float
    tokens_per_second: float


@dataclass
class DecodeResult:
    prefill_tokens: int   # KV cache size at start of decode
    decode_tokens: int
    time_seconds: float
    tokens_per_second: float
    ms_per_token: float


@dataclass
class ProfileRun:
    system: SystemInfo
    prefill: list[PrefillResult]
    decode: list[DecodeResult]

    def save(self, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        safe_model = self.system.model.replace("/", "__").replace(":", "-")
        safe_chip = self.system.chip.replace(" ", "-")
        name = f"profile_{self.system.backend}_{safe_chip}_{safe_model}.json"
        out_path = output_dir / name
        data = {
            "system": asdict(self.system),
            "prefill": [asdict(r) for r in self.prefill],
            "decode": [asdict(r) for r in self.decode],
        }
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)
        return out_path

    @classmethod
    def load(cls, path: Path) -> "ProfileRun":
        with open(path) as f:
            data = json.load(f)
        return cls(
            system=SystemInfo(**data["system"]),
            prefill=[PrefillResult(**r) for r in data["prefill"]],
            decode=[DecodeResult(**r) for r in data["decode"]],
        )
