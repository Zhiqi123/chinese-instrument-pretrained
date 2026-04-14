"""
Fig. 3: Macro-F1 vs trainable parameters (log scale x-axis).
Fig. 4: Macro-F1 vs total method time (s).

Produces:
  $FIG/exp02_perf_vs_params.pdf
  $FIG/exp02_perf_vs_time.pdf
"""

import pathlib
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

FIG_DIR = pathlib.Path(__file__).resolve().parents[2] / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Data from exp02_adaptation_summary_mean_std.csv and
# exp02_adaptation_cost_summary_mean_std.csv (verified)
methods = ["MERT-LP", "LoRA r=4", "LoRA r=8", "Full FT"]
trainable_params = [4614, 152070, 299526, 94376326]
total_time_mean = [159.1, 1278.5, 1612.6, 1359.0]
total_time_std = [11.4, 300.7, 717.6, 543.3]

test_f1_mean = [0.97, 0.76, 0.76, 0.98]
test_f1_std = [0.03, 0.13, 0.09, 0.01]
ext_f1_mean = [0.40, 0.35, 0.40, 0.28]
ext_f1_std = [0.01, 0.18, 0.08, 0.14]

markers = ["o", "s", "D", "^"]
color_test = "#4472C4"
color_ext = "#ED7D31"


def _scatter_plot(ax, x_vals, x_err, test_means, test_stds,
                  ext_means, ext_stds, labels, x_label, log_x=False,
                  legend_loc="upper center", legend_anchor=(0.21, 1)):
    """Plot test and external_test Macro-F1 as two series.

    Each method gets a unique marker shape.  Colour encodes the split
    (blue = test, orange = external test).  The legend shows both
    dimensions so no in-plot text is needed.
    """
    # Plot each method with its own marker
    for i, label in enumerate(labels):
        # Test series
        ax.errorbar(
            x_vals[i], test_means[i], yerr=test_stds[i],
            xerr=x_err[i] if x_err else None,
            fmt=markers[i], color=color_test, markersize=8,
            capsize=3, markeredgecolor="white", markeredgewidth=0.6,
            zorder=3, label=f"{label} (test)",
        )
        # External test series
        ax.errorbar(
            x_vals[i], ext_means[i], yerr=ext_stds[i],
            xerr=x_err[i] if x_err else None,
            fmt=markers[i], color=color_ext, markersize=8,
            capsize=3, markeredgecolor="white", markeredgewidth=0.6,
            zorder=3, label=f"{label} (ext)",
        )

    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Macro-F1")
    ax.set_ylim(0.0, 1.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Build a two-column legend:
    #   Row header — colour patches for "Test" / "External test"
    #   Then one row per method — marker shape only
    from matplotlib.lines import Line2D
    handles, legend_labels = [], []

    # Colour legend (series)
    handles.append(Line2D([], [], color=color_test, marker="o",
                          linestyle="None", markersize=6, label="Test"))
    legend_labels.append("Test")
    handles.append(Line2D([], [], color=color_ext, marker="o",
                          linestyle="None", markersize=6, label="External test"))
    legend_labels.append("External test")

    # Separator — empty handle
    handles.append(Line2D([], [], linestyle="None", label=""))
    legend_labels.append("")

    # Marker legend (methods) — use gray so it's clearly about shape
    for i, label in enumerate(labels):
        handles.append(Line2D([], [], color="gray", marker=markers[i],
                              linestyle="None", markersize=6,
                              markeredgecolor="white", markeredgewidth=0.4))
        legend_labels.append(label)

    ax.legend(handles, legend_labels, loc=legend_loc,
              bbox_to_anchor=legend_anchor,
              frameon=True, edgecolor="gray", ncol=1,
              handletextpad=0.4, labelspacing=0.35)


# --- Fig. 3: Performance vs Trainable Parameters ---
fig3, ax3 = plt.subplots(figsize=(5.5, 3.8))
_scatter_plot(
    ax3, trainable_params, None,
    test_f1_mean, test_f1_std, ext_f1_mean, ext_f1_std,
    methods, "Trainable parameters", log_x=True,
    legend_loc="upper center", legend_anchor=(0.7, 1),
)
plt.tight_layout()
out3 = FIG_DIR / "exp02_perf_vs_params.pdf"
fig3.savefig(out3)
print(f"Saved: {out3}")
plt.close(fig3)

# --- Fig. 4: Performance vs Total Method Time ---
fig4, ax4 = plt.subplots(figsize=(5.5, 3.8))
_scatter_plot(
    ax4, total_time_mean, total_time_std,
    test_f1_mean, test_f1_std, ext_f1_mean, ext_f1_std,
    methods, "Total method time (s)", log_x=False,
    legend_loc="upper center", legend_anchor=(0.19, 1),
)
plt.tight_layout()
out4 = FIG_DIR / "exp02_perf_vs_time.pdf"
fig4.savefig(out4)
print(f"Saved: {out4}")
plt.close(fig4)
