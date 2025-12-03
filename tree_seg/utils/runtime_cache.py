"""Lightweight runtime cache to estimate progress for benchmarks."""

import json
from pathlib import Path
from typing import Optional

from tree_seg.core.types import Config


class RuntimeCache:
    def __init__(self, cache_path: Path | str = ".cache/runtime_estimates.json") -> None:
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._data = self._load()
        # Ensure new structure keys exist
        self._data.setdefault("mean_per_sample", {})
        self._data.setdefault("run_totals", {})

    def _load(self) -> dict:
        if not self.cache_path.exists():
            return {}
        try:
            with self.cache_path.open("r") as f:
                data = json.load(f)
                # Backward-compat: old format was flat dict of mean per sample
                if not isinstance(data, dict) or "mean_per_sample" not in data:
                    data = {"mean_per_sample": data, "run_totals": {}}
                return data
        except Exception:
            return {}

    def _save(self) -> None:
        try:
            with self.cache_path.open("w") as f:
                json.dump(self._data, f, indent=2)
        except Exception:
            pass

    def make_key(self, config: Config) -> str:
        return "/".join(
            [
                config.version,
                config.model_display_name,
                f"stride{config.stride}",
                f"tiling{'on' if config.use_tiling else 'off'}",
                f"img{config.image_size}",
                f"refine{config.refine or 'none'}",
            ]
        )

    def get_mean_runtime(self, key: str) -> Optional[float]:
        value = self._data["mean_per_sample"].get(key)
        return float(value) if value is not None else None

    def update(self, key: str, mean_runtime: float) -> None:
        self._data["mean_per_sample"][key] = float(mean_runtime)
        self._save()

    def make_run_key(self, dataset_name: str, config: Config, num_samples: int) -> str:
        return "/".join(
            [
                dataset_name,
                self.make_key(config),
                f"samples{num_samples}",
            ]
        )

    def get_total_runtime(self, key: str) -> Optional[float]:
        value = self._data["run_totals"].get(key)
        return float(value) if value is not None else None

    def update_total(self, key: str, total_runtime: float) -> None:
        self._data["run_totals"][key] = float(total_runtime)
        self._save()
