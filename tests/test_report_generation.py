"""Verify that measured CSV outputs produce report tables, macros, and figures."""

import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image


def test_report_fragments_are_generated_from_measured_outputs(tmp_path: Path):
    tables = tmp_path / "tables"
    report = tmp_path / "report"
    figures = tmp_path / "figures"
    replication_tables = tmp_path / "replication_tables"
    tables.mkdir()
    figures.mkdir()
    replication_tables.mkdir()
    names = (
        "E0_mobilenetv2_noaug_frozen",
        "E1_mobilenetv2_aug_frozen",
        "E2_convnext_tiny_aug_frozen",
        "E3_convnext_tiny_aug_laststage",
        "E4_convnext_tiny_se_aug_laststage",
    )
    results = pd.DataFrame(
        [
            {
                "exp_name": name,
                "val_f1": 0.70 + index * 0.01,
                "test_accuracy": 0.71 + index * 0.01,
                "test_precision": 0.72 + index * 0.01,
                "test_recall": 0.73 + index * 0.01,
                "test_f1": 0.74 + index * 0.01,
                "test_roc_auc": 0.75 + index * 0.01,
            }
            for index, name in enumerate(names)
        ]
    )
    results.to_csv(tables / "experiment_results.csv", index=False)
    best = names[-1]
    pd.DataFrame(
        [
            {
                "condition": "clean",
                "value": 1.0,
                "accuracy": 0.8,
                "precision": 0.8,
                "recall": 0.8,
                "f1": 0.8,
                "roc_auc": 0.9,
            },
            {
                "condition": "jpeg",
                "value": 20,
                "accuracy": 0.7,
                "precision": 0.7,
                "recall": 0.7,
                "f1": 0.7,
                "roc_auc": 0.8,
            },
            {
                "condition": "brightness",
                "value": 1.3,
                "accuracy": 0.75,
                "precision": 0.75,
                "recall": 0.75,
                "f1": 0.75,
                "roc_auc": 0.85,
            },
            {
                "condition": "noise",
                "value": 0.08,
                "accuracy": 0.65,
                "precision": 0.65,
                "recall": 0.65,
                "f1": 0.65,
                "roc_auc": 0.75,
            },
        ]
    ).to_csv(tables / f"{best}_robustness.csv", index=False)
    pd.DataFrame(
        [
            {
                "source_label": "Human",
                "count": 1000,
                "accuracy": 0.8,
                "error_type": "false_positive_rate",
                "error_rate": 0.2,
            }
        ]
    ).to_csv(tables / f"{best}_source_errors.csv", index=False)
    pd.DataFrame(
        [
            {
                "style_label": "Baroque",
                "count": 200,
                "accuracy": 0.8,
                "precision": 0.8,
                "recall": 0.8,
                "f1": 0.8,
                "roc_auc": 0.9,
            }
        ]
    ).to_csv(tables / f"{best}_style_metrics.csv", index=False)
    for split, count in (("train", 6400), ("val", 1600), ("test", 2000)):
        pd.DataFrame({"row": range(count)}).to_csv(
            tables / f"{split}_split.csv", index=False
        )
    pd.DataFrame(
        [
            {
                "exp_name": name,
                "train_clean_f1": 0.90 + index * 0.01,
                "original_test_f1": 0.88 + index * 0.01,
                "replication_f1": 0.87 + index * 0.01,
                "replication_roc_auc": 0.95 + index * 0.005,
                "replication_f1_ci_low": 0.85 + index * 0.01,
                "replication_f1_ci_high": 0.89 + index * 0.01,
                "train_replication_f1_gap": 0.03,
                "train_replication_f1_gap_ci_low": 0.01,
                "train_replication_f1_gap_ci_high": 0.05,
                "replication_original_f1_delta": -0.01,
            }
            for index, name in enumerate(names)
        ]
    ).to_csv(
        replication_tables / "replication_overfitting_results.csv",
        index=False,
    )
    for suffix in ("confusion", "roc", "robustness_f1"):
        Image.new("RGB", (32, 32), color="white").save(
            figures / f"{best}_{suffix}.png"
        )
    report.mkdir()
    shutil.copyfile("report/report.tex", report / "report.tex")
    shutil.copyfile("report/references.bib", report / "references.bib")

    subprocess.run(
        [
            sys.executable,
            "scripts/make_latex_table.py",
            "--tables-dir",
            str(tables),
            "--output-dir",
            str(report),
            "--figures-dir",
            str(figures),
            "--replication-tables-dir",
            str(replication_tables),
        ],
        check=True,
    )
    assert (report / "generated_results_table.tex").exists()
    assert (report / "generated_robustness_table.tex").exists()
    assert (report / "generated_replication_table.tex").exists()
    robustness_table = (report / "generated_robustness_table.tex").read_text(
        encoding="utf-8"
    )
    assert "Brightness" in robustness_table
    assert "sigma=0.08" in robustness_table
    macros = (report / "generated_metrics.tex").read_text(encoding="utf-8")
    assert r"\newcommand{\TrainCount}{6,400}" in macros
    assert r"\newcommand{\BestTestFOne}{0.780}" in macros
    assert r"\newcommand{\ReplicationCount}{2,000}" in macros
    tectonic = shutil.which("tectonic")
    if tectonic is None:
        pytest.skip("tectonic is not installed")
    subprocess.run(
        [tectonic, "report.tex"],
        cwd=report,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert (report / "report.pdf").exists()


def test_robustness_example_script_generates_panel(tmp_path: Path):
    image_path = tmp_path / "example.png"
    Image.new("RGB", (48, 64), color=(120, 80, 40)).save(image_path)
    split_csv = tmp_path / "split.csv"
    pd.DataFrame(
        [
            {
                "image_path": str(image_path),
                "source_label": "Human",
                "style_label": "Realism",
            }
        ]
    ).to_csv(split_csv, index=False)
    output = tmp_path / "panel.png"

    subprocess.run(
        [
            sys.executable,
            "scripts/make_robustness_examples.py",
            "--split-csv",
            str(split_csv),
            "--output",
            str(output),
            "--image-size",
            "32",
        ],
        check=True,
    )
    assert output.exists()
    with Image.open(output) as panel:
        assert panel.width > panel.height
