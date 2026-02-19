import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Import data
sheet_name = 'major_delay'
df = pd.read_excel('D:/Paper Publish/MARL_TSC/total_result_bus_far_r.xlsx', sheet_name=sheet_name)

col_names = ['Bus', 'Car', 'Person']
controllers = ['PSC', 'ASC', 'ATSP', 'MP', 'MP-TSP', 'LQF', 'LQF-TSP', 'MAPPO', 'MAPPO-MD']
col_num = len(col_names)
xticks_num = len(controllers)
demand = ['Peak', 'Off-peak']

fig, axes = plt.subplots(2, 1, sharex='none', gridspec_kw={'hspace': 0.2}, figsize=(8, 10))
ax1, ax2 = axes.flatten()
axes_lst = [ax1, ax2]


for i in range(2):
    a = list(range(i*col_num*xticks_num, (i+1)*col_num*xticks_num-col_num+1, col_num))
    df0 = df.iloc[:, a]
    df1 = df.iloc[:, [j + 1 for j in a]]
    df2 = df.iloc[:, [j + 2 for j in a]]
    data = pd.concat(
        [df0, pd.DataFrame(df1.values, columns=df0.columns), pd.DataFrame(df2.values, columns=df0.columns)],
        keys=col_names)

    # stack: transfer from DataFrame to Series
    data = data.stack()

    # Add column names
    data = data.rename_axis(index=['Vehicle Type', '', 'Controllers'])

    # Transfer from Series to DataFrame
    data = data.reset_index(level=[0, 2], name='Delay')
    # Plot box plot
    ax = axes_lst[i]
    sns.boxplot(data=data, ax=ax, x='Controllers', hue='Vehicle Type', y='Delay', linewidth=0.2, fliersize=1)
    ax.set_xticks(list(range(xticks_num)), controllers)
    ax.set_xlabel(None)
    ax.set_ylabel('Delay (s)')
    ax.set_title(demand[i])
    ax.set_axisbelow(True)  # Set the axis below the graph element
    ax.yaxis.grid(linestyle='dashed')
    # Add baseline and baseline legend
    handles, labels = ax.get_legend_handles_labels()
    # l_base = ax.axhline(baselines[i], color='red', linestyle='dashed', label='Base Line', zorder=-1)
    # handles.append(l_base)
    ax.legend(loc='upper right')

plt.savefig('Delay statistics in different controllers', dpi=300, bbox_inches='tight')
plt.show()
