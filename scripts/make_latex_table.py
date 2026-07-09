#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd


def write_table(
    frame: pd.DataFrame,
    output_path: Path,
    *,
    column_format: str,
) -> None:
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
    return (
        value.replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("#", r"\#")
    )


def main() -> None:
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

    robustness = pd.read_csv(args.tables_dir / f"{best_name}_robustness.csv")
    robustness_display = robustness[
        ["condition", "value", "accuracy", "precision", "recall", "f1", "roc_auc"]
    ].copy()
    robustness_display["condition"] = robustness_display["condition"].replace(
        {
            "clean": "Clean",
            "contrast": "Contrast",
            "jpeg": "JPEG",
            "resample": "Resample",
        }
    )
    robustness_display["value"] = [
        (
            "--"
            if condition == "clean"
            else (
                f"{value:.1f}x"
                if condition == "contrast"
                else (
                    f"Q{int(value)}"
                    if condition == "jpeg"
                    else f"{int(value)} px"
                )
            )
        )
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

    split_counts = {
        split: len(pd.read_csv(args.tables_dir / f"{split}_split.csv"))
        for split in ("train", "val", "test")
    }
    by_name = results.set_index("exp_name")

    def f1_delta(new: str, baseline: str) -> float:
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
        rf"\newcommand{{\WorstRobustnessCondition}}{{{latex_escape(str(worst_robustness['condition']).upper() if worst_robustness['condition'] == 'jpeg' else str(worst_robustness['condition']).title())}}}",
        rf"\newcommand{{\WorstRobustnessValue}}{{{worst_robustness['value']:g}}}",
        rf"\newcommand{{\WorstRobustnessDelta}}{{{float(worst_robustness['f1']) - clean_f1:+.3f}}}",
        rf"\newcommand{{\HighestErrorSource}}{{{latex_escape(str(highest_source_error['source_label']).replace('_', ' '))}}}",
        rf"\newcommand{{\HighestSourceError}}{{{highest_source_error['error_rate']:.3f}}}",
        rf"\newcommand{{\LowestStyle}}{{{latex_escape(str(lowest_style['style_label']).replace('_', ' '))}}}",
        rf"\newcommand{{\LowestStyleFOne}}{{{lowest_style['f1']:.3f}}}",
        rf"\newcommand{{\HighestStyle}}{{{latex_escape(str(highest_style['style_label']).replace('_', ' '))}}}",
        rf"\newcommand{{\HighestStyleFOne}}{{{highest_style['f1']:.3f}}}",
    ]

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
