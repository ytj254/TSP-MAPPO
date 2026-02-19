import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Sample data
data = {
    'MAPPO_MD': {'Bus': [254.45, 159.66, 108.59, 106.53, 107.02],
                 'Car': [61.35, 63.06, 66.69, 66.85, 66.52],
                 'Person': [61.88, 65.76, 70.04, 71.86, 73.34]},
    'MAPPO': {'Bus': [219.34, 141.97, 109.31, 109.69, 107.02],
              'Car': [63.01, 66.97, 70.55, 70.97, 66.52],
              'Person': [63.45, 69.07, 73.64, 75.86, 73.34]}
}

# Convert the data to a DataFrame
df = pd.DataFrame(data)


# Set font family and size globally for all text elements
plt.rcParams.update({'font.size': 12})

# Plotting the grouped bar plot
fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
controllers = ['MAPPO_MD', 'MAPPO']
bar_width = 0.2  # Adjust the bar width as needed

occupancy_lst = ['1', '10', '30', '50', '70']
headway_lst = ['2 min', '5 min', '10 min', '15 min', '30 min']

for i, controller in enumerate(controllers):
    for idx, (vehicle, values) in enumerate(df[controller].items()):
        x_positions = [pos + idx * bar_width for pos in range(len(values))]
        axes[i].bar(x_positions, values, width=bar_width, label=vehicle)

    axes[i].set_xlabel('Passenger Occupancy')
    axes[i].set_ylabel('Delay (s)')
    axes[i].set_title(controller)
    axes[i].set_xticks(np.arange(len(headway_lst)) + (len(controllers) - 1) * bar_width / 2)
    axes[i].set_xticklabels(headway_lst)
    axes[i].legend()

# Set font for title
# fig.suptitle('Delay Statistics for Different Controllers')

# Save the figure
plt.savefig('Headway_Grouped_Delay_Statistics.png', dpi=300, bbox_inches='tight')
plt.show()
