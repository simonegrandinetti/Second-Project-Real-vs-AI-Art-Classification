from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ProjectConfig:
    """Shared experiment settings.

    The notebook, scripts, and tests all pass this object around instead of
    repeating paths and hyperparameters.  Defaults encode the agreed coursework
    protocol: 224x224 images, batch size 32, eight epochs, and seed 42.
    """

    data_root: Path = Path("data/raw/real-ai-art")
    output_dir: Path = Path("outputs")
    dataset_handle: str = "ravidussilva/real-ai-art/versions/5"
    protocol: str = "official_coursework_10k"
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 4
    epochs: int = 8
    lr_head: float = 1e-3
    lr_finetune: float = 1e-4
    weight_decay: float = 1e-4
    patience: int = 3
    threshold: float = 0.5
    seed: int = 42

    def make_output_dirs(self) -> None:
        """Create the standard output folders if they do not already exist."""
        for name in ("figures", "models", "metrics", "tables"):
            (self.output_dir / name).mkdir(parents=True, exist_ok=True)

    def as_serializable_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation for manifests/checkpoints."""
        values = asdict(self)
        values["data_root"] = str(self.data_root)
        values["output_dir"] = str(self.output_dir)
        return values
