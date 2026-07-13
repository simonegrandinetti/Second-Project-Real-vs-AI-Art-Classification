#!/usr/bin/env python3
"""Generate report fragments from measured experiment outputs.

The report should compile only from real CSV/figure outputs, not hand-entered
numbers. This script reads the experiment tables, creates small LaTeX fragments,
copies selected figures, and writes macros used by `report/report.tex`.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

from ai_art_detection.evaluation import (
    robustness_condition_label,
    robustness_value_label,
)


def write_table(
    frame: pd.DataFrame,
    output_path: Path,
    *,
    column_format: str,
) -> None:
    """Write a DataFrame as a consistently formatted LaTeX table.

    Args:
        frame: Display-ready rows and column headings.
        output_path: Destination ``.tex`` fragment. Its parent must already exist.
        column_format: LaTeX alignment string passed to ``DataFrame.to_latex``.
    """
    output_path.write_text(
        frame.to_latex(
            index=False,
            float_format=lambda value: f"{value:.3f}",
            escape=True,
            column_format=column_format,
        ),
        encoding="utf-8",
    )


def latex_escape(value: str) -> str:
    """Escape special LaTeX characters in text inserted into report macros.

    Args:
        value: Untrusted label from a measured CSV or experiment identifier.

    Returns:
        Text safe to place inside a LaTeX command argument for the characters used by
        this project's generated labels.
    """
    return (
        value.replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("#", r"\#")
    )


def main() -> None:
    """Generate report tables, measured-value macros, and selected figure copies.

    The validation-ranked experiment CSV determines which model supplies robustness,
    subgroup, and diagnostic artifacts. Numerical prose values are written as LaTeX
    macros so the report cannot silently diverge from the measured CSV files. The
    replication table is included only when its independent audit output exists.

    Raises:
        FileNotFoundError: If a required primary result table is absent.
        KeyError: If measured tables do not contain the expected experiment or metric
            columns.
    """
    # 1) Read the locations of measured outputs and report fragments.
    parser = argparse.ArgumentParser(
        description="Generate all LaTeX fragments from measured experiment outputs."
    )
    parser.add_argument(
        "--tables-dir", type=Path, default=Path("outputs/tables")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("report"))
    parser.add_argument(
        "--figures-dir", type=Path, default=Path("outputs/figures")
    )
    parser.add_argument(
        "--replication-tables-dir",
        type=Path,
        default=Path("outputs/replication/tables"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 2) Load the validation-ranked experiment table.  The first row is the
    # model selected for detailed plots, subgroup tables, and robustness.
    results = pd.read_csv(args.tables_dir / "experiment_results.csv")
    results = results.sort_values("val_f1", ascending=False).reset_index(drop=True)
    best = results.iloc[0]
    best_name = str(best["exp_name"])

    result_display = results[
        [
            "exp_name",
            "val_f1",
            "test_accuracy",
            "test_precision",
            "test_recall",
            "test_f1",
            "test_roc_auc",
        ]
    ].copy()
    experiment_labels = {
        "E0_mobilenetv2_noaug_frozen": "E0 MobileNetV2, no aug.",
        "E1_mobilenetv2_aug_frozen": "E1 MobileNetV2, aug.",
        "E2_convnext_tiny_aug_frozen": "E2 ConvNeXt, frozen",
        "E3_convnext_tiny_aug_laststage": "E3 ConvNeXt, last stage",
        "E4_convnext_tiny_se_aug_laststage": "E4 ConvNeXt-SE, last stage",
    }

    # 3) Write the main experiment comparison table.
    result_display["exp_name"] = result_display["exp_name"].map(experiment_labels)
    result_display.columns = [
        "Experiment",
        "Val. F1",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC-AUC",
    ]
    write_table(
        result_display,
        args.output_dir / "generated_results_table.tex",
        column_format="lrrrrrr",
    )

    # 4) Write robustness labels explicitly so the report distinguishes all
    # photometric, compression, blur/noise, and resampling conditions.
    robustness = pd.read_csv(args.tables_dir / f"{best_name}_robustness.csv")
    robustness_display = robustness[
        ["condition", "value", "accuracy", "precision", "recall", "f1", "roc_auc"]
    ].copy()
    robustness_display["condition"] = [
        robustness_condition_label(condition)
        for condition in robustness["condition"]
    ]
    robustness_display["value"] = [
        robustness_value_label(condition, value)
        for condition, value in zip(
            robustness["condition"], robustness["value"]
        )
    ]
    robustness_display.columns = [
        "Condition",
        "Value",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC-AUC",
    ]
    write_table(
        robustness_display,
        args.output_dir / "generated_robustness_table.tex",
        column_format="llrrrrr",
    )

    # 5) Write subgroup tables. Source rows are single-class groups, so the
    # error column is source-specific false-positive or false-negative rate.
    source = pd.read_csv(args.tables_dir / f"{best_name}_source_errors.csv")
    source_display = source[
        ["source_label", "count", "accuracy", "error_type", "error_rate"]
    ].copy()
    source_display["source_label"] = source_display["source_label"].str.replace(
        "_", " ", regex=False
    )
    source_display["error_type"] = source_display["error_type"].replace(
        {
            "false_positive_rate": "False positive",
            "false_negative_rate": "False negative",
        }
    )
    source_display.columns = [
        "Source",
        "Count",
        "Accuracy",
        "Error type",
        "Error rate",
    ]
    write_table(
        source_display,
        args.output_dir / "generated_source_table.tex",
        column_format="lrrlr",
    )

    # Style rows contain both real and fake images, so full binary metrics are
    # meaningful for each style.
    style = pd.read_csv(args.tables_dir / f"{best_name}_style_metrics.csv")
    style_display = style[
        ["style_label", "count", "accuracy", "precision", "recall", "f1", "roc_auc"]
    ].copy()
    style_display["style_label"] = style_display["style_label"].str.replace(
        "_", " ", regex=False
    )
    style_display.columns = [
        "Style",
        "Count",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC-AUC",
    ]
    write_table(
        style_display,
        args.output_dir / "generated_style_table.tex",
        column_format="lrrrrrr",
    )

    # 6) Create macros used in prose.  These keep the report synchronized with
    # the measured CSV files and avoid hand-copying numbers.
    split_counts = {
        split: len(pd.read_csv(args.tables_dir / f"{split}_split.csv"))
        for split in ("train", "val", "test")
    }
    by_name = results.set_index("exp_name")

    def f1_delta(new: str, baseline: str) -> float:
        """Return the signed test-F1 difference between two named experiments."""
        return float(by_name.loc[new, "test_f1"] - by_name.loc[baseline, "test_f1"])

    clean_f1 = float(
        robustness.loc[robustness["condition"] == "clean", "f1"].iloc[0]
    )
    worst_robustness = robustness.loc[robustness["f1"].idxmin()]
    highest_source_error = source.loc[source["error_rate"].idxmax()]
    lowest_style = style.loc[style["f1"].idxmin()]
    highest_style = style.loc[style["f1"].idxmax()]
    macros = [
        rf"\newcommand{{\BestExperiment}}{{{latex_escape(best_name)}}}",
        rf"\newcommand{{\BestTestAccuracy}}{{{best['test_accuracy']:.3f}}}",
        rf"\newcommand{{\BestTestFOne}}{{{best['test_f1']:.3f}}}",
        rf"\newcommand{{\BestTestAUC}}{{{best['test_roc_auc']:.3f}}}",
        rf"\newcommand{{\TrainCount}}{{{split_counts['train']:,}}}",
        rf"\newcommand{{\ValidationCount}}{{{split_counts['val']:,}}}",
        rf"\newcommand{{\TestCount}}{{{split_counts['test']:,}}}",
        rf"\newcommand{{\AugmentationDelta}}{{{f1_delta('E1_mobilenetv2_aug_frozen', 'E0_mobilenetv2_noaug_frozen'):+.3f}}}",
        rf"\newcommand{{\BackboneDelta}}{{{f1_delta('E2_convnext_tiny_aug_frozen', 'E1_mobilenetv2_aug_frozen'):+.3f}}}",
        rf"\newcommand{{\FinetuningDelta}}{{{f1_delta('E3_convnext_tiny_aug_laststage', 'E2_convnext_tiny_aug_frozen'):+.3f}}}",
        rf"\newcommand{{\AttentionDelta}}{{{f1_delta('E4_convnext_tiny_se_aug_laststage', 'E3_convnext_tiny_aug_laststage'):+.3f}}}",
        rf"\newcommand{{\WorstRobustnessCondition}}{{{latex_escape(robustness_condition_label(str(worst_robustness['condition'])))}}}",
        rf"\newcommand{{\WorstRobustnessValue}}{{{latex_escape(robustness_value_label(str(worst_robustness['condition']), float(worst_robustness['value'])))}}}",
        rf"\newcommand{{\WorstRobustnessDelta}}{{{float(worst_robustness['f1']) - clean_f1:+.3f}}}",
        rf"\newcommand{{\HighestErrorSource}}{{{latex_escape(str(highest_source_error['source_label']).replace('_', ' '))}}}",
        rf"\newcommand{{\HighestSourceError}}{{{highest_source_error['error_rate']:.3f}}}",
        rf"\newcommand{{\LowestStyle}}{{{latex_escape(str(lowest_style['style_label']).replace('_', ' '))}}}",
        rf"\newcommand{{\LowestStyleFOne}}{{{lowest_style['f1']:.3f}}}",
        rf"\newcommand{{\HighestStyle}}{{{latex_escape(str(highest_style['style_label']).replace('_', ' '))}}}",
        rf"\newcommand{{\HighestStyleFOne}}{{{highest_style['f1']:.3f}}}",
    ]

    # 7) Add the optional replication audit table/macros when that independent
    # post-training audit has been run.
    replication_path = (
        args.replication_tables_dir / "replication_overfitting_results.csv"
    )
    if replication_path.exists():
        replication = pd.read_csv(replication_path)
        replication_display = replication[
            [
                "exp_name",
                "train_clean_f1",
                "original_test_f1",
                "replication_f1",
                "replication_f1_ci_low",
                "replication_f1_ci_high",
                "train_replication_f1_gap",
                "replication_original_f1_delta",
            ]
        ].copy()
        replication_display["exp_name"] = replication_display["exp_name"].map(
            experiment_labels
        )
        replication_display["replication_ci"] = [
            f"[{low:.3f}, {high:.3f}]"
            for low, high in zip(
                replication_display["replication_f1_ci_low"],
                replication_display["replication_f1_ci_high"],
            )
        ]
        replication_display = replication_display[
            [
                "exp_name",
                "train_clean_f1",
                "original_test_f1",
                "replication_f1",
                "replication_ci",
                "train_replication_f1_gap",
                "replication_original_f1_delta",
            ]
        ]
        replication_display.columns = [
            "Experiment",
            "Train F1",
            "Test-1 F1",
            "Test-2 F1",
            "Test-2 95% CI",
            "Train--Test-2",
            "Test-2--Test-1",
        ]
        write_table(
            replication_display,
            args.output_dir / "generated_replication_table.tex",
            column_format="lrrrlrr",
        )
        selected = replication.set_index("exp_name").loc[
            "E3_convnext_tiny_aug_laststage"
        ]
        replication_best = replication.loc[replication["replication_f1"].idxmax()]
        macros.extend(
            [
                r"\newcommand{\ReplicationCount}{2,000}",
                rf"\newcommand{{\ReplicationBestExperiment}}{{{latex_escape(experiment_labels[str(replication_best['exp_name'])])}}}",
                rf"\newcommand{{\SelectedReplicationFOne}}{{{selected['replication_f1']:.3f}}}",
                rf"\newcommand{{\SelectedReplicationFOneCILow}}{{{selected['replication_f1_ci_low']:.3f}}}",
                rf"\newcommand{{\SelectedReplicationFOneCIHigh}}{{{selected['replication_f1_ci_high']:.3f}}}",
                rf"\newcommand{{\SelectedTrainReplicationGap}}{{{selected['train_replication_f1_gap']:+.3f}}}",
                rf"\newcommand{{\SelectedGapCILow}}{{{selected['train_replication_f1_gap_ci_low']:+.3f}}}",
                rf"\newcommand{{\SelectedGapCIHigh}}{{{selected['train_replication_f1_gap_ci_high']:+.3f}}}",
                rf"\newcommand{{\SelectedReplicationOriginalDelta}}{{{selected['replication_original_f1_delta']:+.3f}}}",
            ]
        )
    (args.output_dir / "generated_metrics.tex").write_text(
        "\n".join(macros) + "\n", encoding="utf-8"
    )

    # 8) Copy selected measured figures into report-friendly names.
    figure_sources = {
        f"{best_name}_confusion.png": "generated_confusion.png",
        f"{best_name}_roc.png": "generated_roc.png",
        f"{best_name}_robustness_f1.png": "generated_robustness.png",
        "gradcam_correct_real.png": "generated_gradcam_correct_real.png",
        "gradcam_correct_fake.png": "generated_gradcam_correct_fake.png",
        "gradcam_misclassified.png": "generated_gradcam_misclassified.png",
    }
    for source_name, destination_name in figure_sources.items():
        source_path = args.figures_dir / source_name
        if source_path.exists():
            shutil.copyfile(source_path, args.output_dir / destination_name)
    print(f"Generated report fragments for {best_name} in {args.output_dir}")


if __name__ == "__main__":
    main()
