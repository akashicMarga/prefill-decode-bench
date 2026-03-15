"""
Shared dataclasses. Both backends produce these — plot_results.py
and the summary printer consume them without knowing which backend ran.
"""

from dataclasses import dataclass, asdict, field
import json
from pathlib import Path


APPLE_BANDWIDTH_GBS = {
    "Apple M1": 68.25,
    "Apple M1 Pro": 200.0,
    "Apple M1 Max": 400.0,
    "Apple M1 Ultra": 800.0,
    "Apple M2": 100.0,
    "Apple M2 Pro": 200.0,
    "Apple M2 Max": 400.0,
    "Apple M2 Ultra": 800.0,
    "Apple M3": 100.0,
    "Apple M3 Pro": 150.0,
    "Apple M3 Max": 400.0,
    "Apple M3 Ultra": 800.0,
    "Apple M4": 120.0,
    "Apple M4 Pro": 273.0,
    "Apple M4 Max": 546.0,
    "Apple M5": 120.0,
    "Apple M5 Pro": 273.0,
    "Apple M5 Max": 614.0,
}


def lookup_bandwidth(chip: str) -> float | None:
    for name, bw in APPLE_BANDWIDTH_GBS.items():
        if name in chip or chip in name:
            return bw
    return None


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
    peak_memory_gb: float = 0.0
    tflops: float = 0.0


@dataclass
class DecodeResult:
    prefill_tokens: int   # KV cache size at start of decode
    decode_tokens: int
    time_seconds: float
    tokens_per_second: float
    ms_per_token: float
    peak_memory_gb: float = 0.0
    effective_bandwidth_gbs: float = 0.0
    bandwidth_utilization_pct: float = 0.0
    arithmetic_intensity: float = 0.0


@dataclass
class HardwareMetrics:
    model_size_gb: float = 0.0
    model_params_b: float = 0.0
    theoretical_bandwidth_gbs: float = 0.0
    model_load_memory_gb: float = 0.0


@dataclass
class ProfileRun:
    system: SystemInfo
    prefill: list[PrefillResult]
    decode: list[DecodeResult]
    hardware: HardwareMetrics | None = None

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
        if self.hardware:
            data["hardware"] = asdict(self.hardware)
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)
        return out_path

    @classmethod
    def load(cls, path: Path) -> "ProfileRun":
        with open(path) as f:
            data = json.load(f)
        hw = None
        if "hardware" in data:
            hw = HardwareMetrics(**data["hardware"])
        return cls(
            system=SystemInfo(**data["system"]),
            prefill=[PrefillResult(**r) for r in data["prefill"]],
            decode=[DecodeResult(**r) for r in data["decode"]],
            hardware=hw,
        )
