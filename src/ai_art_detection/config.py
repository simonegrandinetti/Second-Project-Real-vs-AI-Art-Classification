"""Central configuration for the reproducible coursework workflow.

Keeping the paths and hyperparameters in one dataclass lets the notebook and
command-line entry points use the same defaults without maintaining parallel
configuration dictionaries.
"""

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

    Attributes:
        data_root: Extracted AI-ArtBench directory.
        output_dir: Root directory for generated models, metrics, tables, and figures.
        dataset_handle: Pinned Kaggle dataset identifier recorded in manifests.
        protocol: Human-readable name of the sampling and evaluation protocol.
        image_size: Square input size supplied to every model.
        batch_size: Number of images per optimizer or evaluation step.
        num_workers: Worker processes used by each PyTorch DataLoader.
        epochs: Maximum number of training epochs before early stopping.
        lr_head: AdamW learning rate for classifier-only training.
        lr_finetune: Lower AdamW learning rate used when backbone layers are unfrozen.
        weight_decay: Decoupled weight-decay coefficient passed to AdamW.
        patience: Consecutive non-improving validation epochs allowed before stopping.
        threshold: Fake-class probability threshold used to convert logits to labels.
        seed: Base seed for sampling, augmentation, and model training.
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
        """Create the standard output folders below :attr:`output_dir`.

        Existing folders and files are left untouched, so this method is safe to call
        before both new and resumed experiment runs.
        """
        for name in ("figures", "models", "metrics", "tables"):
            (self.output_dir / name).mkdir(parents=True, exist_ok=True)

    def as_serializable_dict(self) -> dict[str, Any]:
        """Return the configuration in a JSON-friendly form.

        Returns:
            A new dictionary containing every dataclass field, with both ``Path``
            values converted to strings for manifests and checkpoint metadata.
        """
        values = asdict(self)
        values["data_root"] = str(self.data_root)
        values["output_dir"] = str(self.output_dir)
        return values
