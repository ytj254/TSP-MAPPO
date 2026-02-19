import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams.update({
    "font.family": "Times New Roman",   # or "serif" if TNR not available
    "font.size": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
})

# ---------------- config ----------------
CONFIG = {
    # set one of: "real" or "hypo"
    "scenario": "real",

    # input files / baselines / output names
    "real": {
        "excel": r"D:/Paper Publish/MARL_TSC/total_result_bus_far_r.xlsx",
        "baselines": [85.64, 85.64, 72.98, 72.98],
        "out_png": "Delay_statistics_MPR_real.png",
    },
    "hypo": {
        "excel": r"D:/Paper Publish/MARL_TSC/total_result_bus_far.xlsx",
        "baselines": [114.66, 114.66, 56.81, 56.81],
        "out_png": "Delay_statistics_MPR_hypo.png",
    },

    # plotting controls
    "figsize": (10, 8),
    "dpi": 300,
    "showfliers": True,
    "fliersize": 1,
    "linewidth": 0.2,
    "legend_loc": "upper right",
    "baseline_label": "Baseline",
    "baseline_style": {"linestyle": "dashed", "linewidth": 1.0},
    "caption_y": -0.08,          # move panel caption up/down (closer to chart)
    "tight_layout": True,
    "hspace": 0.35,              # vertical space between rows
}

# ---------------- constants ----------------
sheet_name = "mpr_box"
col_names = ["Bus", "Car", "Person"]
mpr_lst = ["20%", "40%", "60%", "80%", "100%"]
controllers = ["(a) Peak: MAPPO-M", "(b) Peak: MAPPO",
               "(c) Off-peak: MAPPO-M", "(d) Off-peak: MAPPO"]

col_num = len(col_names)

def parse_args():
    parser = argparse.ArgumentParser(description="Plot MPR boxplots.")
    # Use --scenario to switch between the configured datasets (real or hypo).
    parser.add_argument(
        "--scenario",
        choices=["real", "hypo"],
        default=CONFIG["scenario"],
        help="Scenario to plot.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ---------------- load data ----------------
    scenario = args.scenario
    excel_path = CONFIG[scenario]["excel"]
    baselines = CONFIG[scenario]["baselines"]
    out_png = CONFIG[scenario]["out_png"]

    df = pd.read_excel(excel_path, sheet_name=sheet_name)

    # ---------------- plot ----------------
    fig, axes = plt.subplots(2, 2, figsize=CONFIG["figsize"])

    for r in range(2):
        for c in range(2):
            i = r * 2 + c
            ax = axes[r, c]

            a = list(range(i * col_num * len(mpr_lst), (i + 1) * col_num * len(mpr_lst) - col_num + 1, col_num))
            df0 = df.iloc[:, a]
            df1 = df.iloc[:, [j + 1 for j in a]]
            df2 = df.iloc[:, [j + 2 for j in a]]

            data = pd.concat(
                [df0,
                 pd.DataFrame(df1.values, columns=df0.columns),
                 pd.DataFrame(df2.values, columns=df0.columns)],
                keys=col_names
            ).stack()

            data = data.rename_axis(index=["Vehicle Type", "", "MPR"])
            data = data.reset_index(level=[0, 2], name="Delay")

            sns.boxplot(
                data=data, ax=ax, x="MPR", hue="Vehicle Type", y="Delay",
                linewidth=CONFIG["linewidth"], fliersize=CONFIG["fliersize"],
                showfliers=CONFIG["showfliers"],
            )

            ax.set_xticks(list(range(len(mpr_lst))), mpr_lst)
            ax.set_xlabel(None)
            # show y tick labels on every subplot (including right column)
            # set ticks to point inward
            ax.tick_params(axis="both", direction="in", labelleft=True, labelright=False, right=False)

            ax.axhline(
                baselines[i], color="red", label=CONFIG["baseline_label"],
                **CONFIG["baseline_style"], zorder=0
            )

            ax.legend(loc=CONFIG["legend_loc"], frameon=False)

            ax.text(0.5, CONFIG["caption_y"], controllers[i],
                    transform=ax.transAxes, ha="center", va="top", fontsize=12)

    # Synchronize y-axis limits for subplots in the same row
    for r in range(2):
        ylims = [axes[r, c].get_ylim() for c in range(2)]
        combined_ylim = (min(y[0] for y in ylims), max(y[1] for y in ylims))
        for c in range(2):
            axes[r, c].set_ylim(combined_ylim)

    if CONFIG["tight_layout"]:
        # Add extra vertical space between rows for controller labels
        fig.subplots_adjust(hspace=CONFIG["hspace"])
        fig.tight_layout()

    # Set y-labels AFTER tight_layout so they appear on all subplots
    for ax in axes.flat:
        ax.set_ylabel("Delay /s", labelpad=8)

    fig.savefig(out_png, dpi=CONFIG["dpi"], bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()