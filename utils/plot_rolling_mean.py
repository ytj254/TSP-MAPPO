import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import ast

# ---- Global style ----
plt.rcParams.update({
    "font.family": "Times New Roman",   # or "serif" if TNR not available
    "font.size": 10,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
})

def space_thousands(x, pos):
    if abs(x - int(x)) < 1e-9:
        return f"{int(x):,}".replace(",", " ")
    return f"{x:g}"

def read_and_process(file_path):
    df = pd.read_csv(file_path)
    # safer than eval
    all_values = df["hist_stats/episode_reward"].apply(
        lambda x: ast.literal_eval(x) if pd.notna(x) else []
    ).tolist()

    unique_values = set()
    result_list = []
    for v in [item for sublist in all_values for item in sublist]:
        if v not in unique_values:
            unique_values.add(v)
            result_list.append(v)
    return pd.Series(result_list)

# ---------------- result paths ----------------
absolute_path = 'D:/Paper Publish/MARL_TSC/MARL_TSC_code'
file_paths_h = [
    f'{absolute_path}/agents/h_bus/850/ippo/PPO_2024-04-25_23-04-36/progress.csv',
    f'{absolute_path}/agents/h_bus/850/ctde/no_md/PPO_2024-02-10_20-17-45/PPO_sumo_env/progress.csv',
    f'{absolute_path}/agents/h_bus/850/ctde/md_far/PPO_2024-02-11_21-57-54/PPO_sumo_env/progress.csv',
    f'{absolute_path}/agents/h_bus/400/ippo/PPO_2024-05-21_10-48-08/progress.csv',
    f'{absolute_path}/agents/h_bus/400/ctde/no_md/PPO_2024-02-24_23-52-13/PPO_sumo_env/progress.csv',
    f'{absolute_path}/agents/h_bus/400/ctde/md_far/PPO_2024-02-24_00-05-18/PPO_sumo_env/progress.csv',
]
file_paths_r = [
    f'{absolute_path}/agents/r_bus/peak/ippo/PPO_2024-05-21_16-16-57/progress.csv',
    f'{absolute_path}/agents/r_bus/peak/ctde/no_md/PPO_2024-02-20_19-04-53/PPO_sumo_env/progress.csv',
    f'{absolute_path}/agents/r_bus/peak/ctde/md_far/PPO_2024-02-18_18-08-28/PPO_sumo_env/progress.csv',
    f'{absolute_path}/agents/r_bus/offpeak/ippo/PPO_2024-05-21_22-18-40/progress.csv',
    f'{absolute_path}/agents/r_bus/offpeak/ctde/no_md/PPO_2024-02-22_23-57-56/PPO_sumo_env/progress.csv',
    f'{absolute_path}/agents/r_bus/offpeak/ctde/md_far/PPO_2024-02-23_18-22-28/PPO_sumo_env/progress.csv',
]

label_names = ['IPPO', 'MAPPO', 'MAPPO-M']
chart_names = ['(a) Hypothetical Peak', '(b) Hypothetical Off-Peak',
               '(c) Real-world Peak', '(d) Real-world Off-Peak']
file_paths_combined = [file_paths_h[:3], file_paths_h[3:], file_paths_r[:3], file_paths_r[3:]]

# ---- Figure size ----
fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.6))

for idx, ax in enumerate(axes.flat):
    data_paths = file_paths_combined[idx]

    for file_path, label_name in zip(data_paths, label_names):
        s = read_and_process(file_path)
        mean = s.rolling(window=50).mean()
        # std = s.rolling(window=50, min_periods=1).std().rolling(window=10, min_periods=1).mean()

        ax.plot(mean.index, mean.values, label=label_name, linewidth=1.2)
        # ax.fill_between(mean.index,
        #                 (mean - 0.75 * std).values,
        #                 (mean + 0.75 * std).values,
        #                 alpha=0.18)

    # axis labels on all panels
    ax.set_xlabel('Episode')
    ax.set_ylabel('Mean episode reward')

    # format ticks: 1 000
    ax.xaxis.set_major_formatter(FuncFormatter(space_thousands))
    ax.yaxis.set_major_formatter(FuncFormatter(space_thousands))

    # inward ticks, top/right ticks optional for journal look
    ax.tick_params(axis='both', which='both', direction='in', length=3, width=0.8)

    # frame line width
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    # legend no frame
    if ax.lines:
        ax.legend(loc='lower right', frameon=False, handlelength=2.2)

    # panel caption below each subplot (not overlapping x ticks)
    ax.text(0.5, -0.22, chart_names[idx],
            transform=ax.transAxes, ha='center', va='top')

# align y-label x-position across rows/cols
fig.align_ylabels(axes[:, 0])
fig.align_ylabels(axes[:, 1])

fig.tight_layout()
# ---- Save: PNG  ----
fig.savefig('mean_episode_reward_curve.png', dpi=300, bbox_inches='tight')
plt.show()
