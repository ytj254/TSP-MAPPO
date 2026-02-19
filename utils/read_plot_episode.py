import pandas as pd
import matplotlib.pyplot as plt


def read_and_process(file_path):
    df = pd.read_csv(file_path)
    all_values = df["hist_stats/episode_reward"].apply(eval).tolist()

    unique_values_set = set()
    result_list = []

    for value in [item for sublist in all_values for item in sublist]:
        if value not in unique_values_set:
            unique_values_set.add(value)
            result_list.append(value)
    return result_list


absolute_path = 'D:/Paper Publish/MARL_TSC/MARL_TSC_code'

file_paths_h = [
    f'{absolute_path}/agents/h_bus/850/ctde/no_md/PPO_2024-02-10_20-17-45/PPO_sumo_env/progress.csv',
    f'{absolute_path}/agents/h_bus/850/ctde/md_far/PPO_2024-02-11_21-57-54/PPO_sumo_env/progress.csv',
    f'{absolute_path}/agents/h_bus/400/ctde/no_md/PPO_2024-02-24_23-52-13/PPO_sumo_env/progress.csv',
    f'{absolute_path}/agents/h_bus/400/ctde/md_far/PPO_2024-02-24_00-05-18/PPO_sumo_env/progress.csv',
]

file_paths_r = [
    f'{absolute_path}/agents/r_bus/peak/ctde/no_md/PPO_2024-02-20_19-04-53/PPO_sumo_env/progress.csv',
    f'{absolute_path}/agents/r_bus/peak/ctde/md_far/PPO_2024-02-18_18-08-28/PPO_sumo_env/progress.csv',
    f'{absolute_path}/agents/r_bus/offpeak/ctde/no_md/PPO_2024-02-22_23-57-56/PPO_sumo_env/progress.csv',
    f'{absolute_path}/agents/r_bus/offpeak/ctde/md_far/PPO_2024-02-23_18-22-28/PPO_sumo_env/progress.csv',
]

label_names = ['MAPPO', 'MAPPO-M']

# Create the figure and axis objects
fig, [ax1, ax2] = plt.subplots(2, 1, sharex='all', gridspec_kw={'hspace': 0.2}, figsize=(8, 7))
ax_lst = [ax1, ax2]
chart_names = ['Peak', 'Off-Peak']
font = {
    # 'family': 'times new roman',
    'color': 'black',
    # 'weight': 'bold',
    'size': 12,
}

for i in range(2):
    data_paths = [file_paths_h[i*2], file_paths_h[i*2+1]]
    ax = ax_lst[i]
    for file_path, label_name in zip(data_paths, label_names):
        result_list = read_and_process(file_path)

        # Plotting the episode reward changing curve
        ax.plot(result_list, label=label_name)

        # Set the axis labels and legend
        ax.set_ylabel('Episode Reward', fontdict=font)
        ax.set_title(chart_names[i], fontdict=font)
        ax.grid(linestyle='dashed')

        # Add legend
        ax.legend()

plt.xlabel('Episode', fontdict=font)

# Save the figure as an image (e.g., PNG) if you want to insert it into a Word document
plt.savefig('episode_reward_curve.png', format='png', dpi=300)

plt.show()
