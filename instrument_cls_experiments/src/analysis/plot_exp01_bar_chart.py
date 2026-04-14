"""
Fig. 2 — Exp 1 test vs external_test Macro-F1 grouped bar chart.

Produces:
  $FIG/exp01_test_vs_ext_macro_f1.pdf
"""

import pathlib
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

FIG_DIR = pathlib.Path(__file__).resolve().parents[2] / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Data from exp01_transfer_summary_mean_std.csv (verified)
methods = ["MFCC+SVM", "CLAP-ZS", "CLAP-LP", "MERT-LP"]
test_f1_mean = [0.9905, 0.0465, 0.9545, 0.9675]
test_f1_std  = [0.0084, 0.0150, 0.0072, 0.0298]
ext_f1_mean  = [0.1169, 0.1059, 0.5906, 0.3959]
ext_f1_std   = [0.0260, 0.0000, 0.0716, 0.0138]

x = np.arange(len(methods))
width = 0.35

fig, ax = plt.subplots(figsize=(5.5, 3.5))

bars_test = ax.bar(
    x - width / 2, test_f1_mean, width,
    yerr=test_f1_std, capsize=3,
    label="Test", color="#4472C4", edgecolor="white", linewidth=0.5,
)
bars_ext = ax.bar(
    x + width / 2, ext_f1_mean, width,
    yerr=ext_f1_std, capsize=3,
    label="External test", color="#ED7D31", edgecolor="white", linewidth=0.5,
)

# Value labels on top of bars (2 decimal places)
# Place label above error bar cap to avoid overlap
for bar, mean, sd in zip(bars_test, test_f1_mean, test_f1_std):
    top = mean + sd
    ax.annotate(
        f"{mean:.2f}",
        xy=(bar.get_x() + bar.get_width() / 2, top),
        xytext=(0, 4),
        textcoords="offset points",
        ha="center", va="bottom",
        fontsize=8,
    )
for bar, mean, sd in zip(bars_ext, ext_f1_mean, ext_f1_std):
    top = mean + sd
    ax.annotate(
        f"{mean:.2f}",
        xy=(bar.get_x() + bar.get_width() / 2, top),
        xytext=(0, 4),
        textcoords="offset points",
        ha="center", va="bottom",
        fontsize=8,
    )

ax.set_ylabel("Macro-F1")
ax.set_ylim(0.0, 1.18)
ax.set_xticks(x)
ax.set_xticklabels(methods)
ax.legend(loc="upper center", bbox_to_anchor=(0.34, 1.0), frameon=True, edgecolor="gray")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
out_path = FIG_DIR / "exp01_test_vs_ext_macro_f1.pdf"
fig.savefig(out_path)
print(f"Saved: {out_path}")
plt.close(fig)
