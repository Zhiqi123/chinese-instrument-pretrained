"""Aggregate Phase 1/2 confusion matrices into summary CSVs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXP_ROOT = PROJECT_ROOT / "instrument_cls_experiments"


PHASE1_METHODS = [
    {"method_short": "mfcc_svm", "method_display": "MFCC+SVM",
     "model": "mfcc_svm", "experiment_id": "exp01_mfcc_svm",
     "source_phase": 1, "runs_subdir": "exp01_transfer"},
    {"method_short": "clap_zeroshot", "method_display": "CLAP zero-shot",
     "model": "clap_htsat_unfused", "experiment_id": "exp01_clap_zeroshot",
     "source_phase": 1, "runs_subdir": "exp01_transfer"},
    {"method_short": "clap_linear", "method_display": "CLAP linear probe",
     "model": "clap_htsat_unfused", "experiment_id": "exp01_clap_linear",
     "source_phase": 1, "runs_subdir": "exp01_transfer"},
    {"method_short": "mert_linear", "method_display": "MERT linear probe",
     "model": "mert_v1_95m", "experiment_id": "exp01_mert_linear",
     "source_phase": 1, "runs_subdir": "exp01_transfer"},
]

PHASE2_METHODS = [
    {"method_short": "mert_linear", "method_display": "MERT linear probe",
     "model": "mert_v1_95m", "experiment_id": "exp02_mert_linear",
     "source_phase": 2, "runs_subdir": "exp02_adaptation"},
    {"method_short": "mert_lora_r4", "method_display": "MERT+LoRA (r=4)",
     "model": "mert_v1_95m", "experiment_id": "exp02_mert_lora_r4",
     "source_phase": 2, "runs_subdir": "exp02_adaptation"},
    {"method_short": "mert_lora_r8", "method_display": "MERT+LoRA (r=8)",
     "model": "mert_v1_95m", "experiment_id": "exp02_mert_lora_r8",
     "source_phase": 2, "runs_subdir": "exp02_adaptation"},
    {"method_short": "mert_full_ft", "method_display": "MERT full fine-tuning",
     "model": "mert_v1_95m", "experiment_id": "exp02_mert_full_ft",
     "source_phase": 2, "runs_subdir": "exp02_adaptation"},
]

SEED_NAMES = ["seed0", "seed1", "seed2"]
SPLITS = ["test", "external_test"]


def collect_confusion_cells(
    methods: list[dict],
    runs_root: Path,
) -> pd.DataFrame:
    """Load all confusion matrix CSVs and expand into cell-level rows."""
    rows: list[dict] = []

    for m in methods:
        for seed in SEED_NAMES:
            for split in SPLITS:
                cm_path = (
                    runs_root / m["runs_subdir"] / m["method_short"]
                    / seed / f"{split}_confusion_matrix.csv"
                )
                if not cm_path.exists():
                    raise FileNotFoundError(
                        f"Confusion matrix missing: {cm_path.relative_to(EXP_ROOT)}"
                    )

                # first col = true_label (index), column headers = pred_label
                cm_df = pd.read_csv(cm_path, index_col=0)

                for true_label in cm_df.index:
                    row_total = int(cm_df.loc[true_label].sum())
                    for pred_label in cm_df.columns:
                        count = int(cm_df.loc[true_label, pred_label])
                        row_rate = count / row_total if row_total > 0 else 0.0
                        rows.append({
                            "source_phase": m["source_phase"],
                            "experiment_id": m["experiment_id"],
                            "method": m["method_display"],
                            "method_short": m["method_short"],
                            "model": m["model"],
                            "seed_name": seed,
                            "split": split,
                            "true_label": true_label,
                            "pred_label": pred_label,
                            "count": count,
                            "row_total": row_total,
                            "row_rate": row_rate,
                        })

    return pd.DataFrame(rows)


def aggregate_confusion_mean_std(cell_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate mean±std(ddof=1) across 3 seeds."""
    group_cols = [
        "source_phase", "experiment_id", "method", "method_short",
        "model", "split", "true_label", "pred_label",
    ]
    agg = cell_df.groupby(group_cols, sort=False).agg(
        count_mean=("count", "mean"),
        count_std=("count", lambda x: x.std(ddof=1)),
        row_rate_mean=("row_rate", "mean"),
        row_rate_std=("row_rate", lambda x: x.std(ddof=1)),
        seed_count=("seed_name", "nunique"),
    ).reset_index()

    return agg


def main() -> None:
    runs_root = EXP_ROOT / "runs"
    output_dir = EXP_ROOT / "reports/tables"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("build_exp04_confusion_summary: collecting Phase 1/2 confusion cells ...")
    all_methods = PHASE1_METHODS + PHASE2_METHODS
    cell_df = collect_confusion_cells(all_methods, runs_root)
    print(f"  cell_df: {len(cell_df)} rows, "
          f"{cell_df['experiment_id'].nunique()} experiments, "
          f"{cell_df['seed_name'].nunique()} seeds")

    cell_path = output_dir / "exp04_confusion_cell_summary.csv"
    cell_df.to_csv(cell_path, index=False)
    print(f"  wrote: {cell_path.relative_to(EXP_ROOT)}")

    print("  aggregating mean±std ...")
    mean_std_df = aggregate_confusion_mean_std(cell_df)
    mean_std_path = output_dir / "exp04_confusion_mean_std.csv"
    mean_std_df.to_csv(mean_std_path, index=False)
    print(f"  wrote: {mean_std_path.relative_to(EXP_ROOT)}")

    print("build_exp04_confusion_summary: done")


if __name__ == "__main__":
    main()
