"""Generate LaTeX-ready layer probe table from summary CSV."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXP_ROOT = PROJECT_ROOT / "instrument_cls_experiments"

TABLES_DIR = EXP_ROOT / "reports/tables"


def _fmt(mean: float, std: float, is_best: bool = False) -> str:
    """Format mean±std; bold if best."""
    s = f"{mean:.4f}±{std:.4f}"
    if is_best:
        s = f"**{s}**"
    return s


def main() -> None:
    print("build_exp04_layer_probe_table: generating LaTeX-ready table")
    print("=" * 60)

    csv_path = TABLES_DIR / "exp04_layer_probe_summary_mean_std.csv"
    if not csv_path.exists():
        print(f"  ERROR: {csv_path} not found")
        sys.exit(1)

    df = pd.read_csv(csv_path)

    # pivot: expand per-split metrics into columns
    test_df = df[df["split"] == "test"].set_index("layer_index")
    ext_df = df[df["split"] == "external_test"].set_index("layer_index")

    # find best layer
    test_best_layer = test_df["macro_f1_mean"].idxmax() if not test_df.empty else -1
    ext_best_layer = ext_df["macro_f1_mean"].idxmax() if not ext_df.empty else -1

    # build table rows
    all_layers = sorted(df["layer_index"].unique())
    rows = []
    for layer in all_layers:
        layer_name = "embedding_output" if layer == 0 else f"transformer_layer_{layer}"
        row = {"layer_index": int(layer), "layer_name": layer_name}

        if layer in test_df.index:
            t = test_df.loc[layer]
            is_best_test = (layer == test_best_layer)
            row["test_macro_f1"] = _fmt(t["macro_f1_mean"], t["macro_f1_std"], is_best_test)
            row["test_accuracy"] = _fmt(t["accuracy_mean"], t["accuracy_std"], is_best_test)
        else:
            row["test_macro_f1"] = "-"
            row["test_accuracy"] = "-"

        if layer in ext_df.index:
            e = ext_df.loc[layer]
            is_best_ext = (layer == ext_best_layer)
            row["ext_macro_f1"] = _fmt(e["macro_f1_mean"], e["macro_f1_std"], is_best_ext)
            row["ext_accuracy"] = _fmt(e["accuracy_mean"], e["accuracy_std"], is_best_ext)
        else:
            row["ext_macro_f1"] = "-"
            row["ext_accuracy"] = "-"

        rows.append(row)

    table_df = pd.DataFrame(rows)

    output_path = TABLES_DIR / "exp04_layer_probe_table.csv"
    table_df.to_csv(output_path, index=False)
    print(f"  wrote: {output_path.relative_to(EXP_ROOT)}")
    print(f"  rows: {len(table_df)}")

    # preview
    print("-" * 60)
    print(table_df.to_string(index=False))

    print("=" * 60)
    print("build_exp04_layer_probe_table: done")


if __name__ == "__main__":
    main()
