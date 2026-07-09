"""Tools for binary detection of AI-generated artwork."""

from .config import ProjectConfig
from .data import (
    ArtBinaryDataset,
    balanced_sample,
    build_standard_eval_loader,
    build_loaders,
    coursework_split,
    replication_test_split,
    scan_dataset,
    sample_source_style_quotas,
    stratified_split,
    validate_dataset_inventory,
    validate_image_readability,
)
from .models import build_model, count_trainable_parameters

__all__ = [
    "ArtBinaryDataset",
    "ProjectConfig",
    "balanced_sample",
    "build_standard_eval_loader",
    "build_loaders",
    "build_model",
    "count_trainable_parameters",
    "coursework_split",
    "replication_test_split",
    "scan_dataset",
    "sample_source_style_quotas",
    "stratified_split",
    "validate_dataset_inventory",
    "validate_image_readability",
]
