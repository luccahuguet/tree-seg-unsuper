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

    def _load(self) -> dict:
        if not self.cache_path.exists():
            return {}
        try:
            with self.cache_path.open("r") as f:
                return json.load(f)
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
        value = self._data.get(key)
        return float(value) if value is not None else None

    def update(self, key: str, mean_runtime: float) -> None:
        self._data[key] = float(mean_runtime)
        self._save()
