import argparse
import pandas as pd
import matplotlib.pyplot as plt

# ---- Global style ----
plt.rcParams.update({
    "font.family": "Times New Roman",   # or "serif" if TNR not available
    "font.size": 12,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
})

# ---------------- Data Configuration ----------------
DATA_CONFIG = {
    "occupancy": {
        "hypo_peak": {
            'MAPPO-M': {'Bus': [254.45, 159.66, 108.59, 106.53, 107.02],
                         'Car': [61.35, 63.06, 66.69, 66.85, 66.52],
                         'Person': [61.88, 65.76, 70.04, 71.86, 73.34]},
            'MAPPO': {'Bus': [219.34, 141.97, 109.31, 109.69, 110.14],
                      'Car': [63.01, 66.97, 70.55, 70.97, 70.29],
                      'Person': [63.45, 69.07, 73.64, 75.86, 76.99]}
        },
        "hypo_offpeak": {
            'MAPPO-M': {'Bus': [177.96, 91.89, 83.81, 83.86, 83.86],
                         'Car': [38.51, 39.84, 40.05, 40.03, 40.03],
                         'Person': [39.36, 42.84, 46.84, 50.30, 53.17]},
            'MAPPO': {'Bus': [182.15, 120.52, 95.68, 95.14, 95.04],
                      'Car': [40.21, 40.65, 40.37, 40.41, 40.55],
                      'Person': [41.07, 45.26, 48.95, 53.25, 56.90]}
        },
        "real_peak": {
            'MAPPO-M': {'Bus': [189.75, 129.49, 102.48, 98.63, 98.11],
                         'Car': [58.99, 60.08, 62.86, 63.16, 63.28],
                         'Person': [59.56, 63.06, 67.56, 69.66, 71.61]},
            'MAPPO': {'Bus': [185.92, 134.95, 114.58, 115.28, 116.80],
                      'Car': [56.47, 57.64, 60.00, 60.66, 61.09],
                      'Person': [57.03, 60.96, 66.48, 70.66, 74.41]}
        },
        "real_offpeak": {
            'MAPPO-M': {'Bus': [161.80, 112.93, 95.58, 92.51, 91.62],
                         'Car': [46.50, 46.91, 48.03, 48.21, 48.44],
                         'Person': [47.28, 51.19, 56.22, 59.61, 62.54]},
            'MAPPO': {'Bus': [182.84, 115.09, 100.21, 98.94, 98.97],
                      'Car': [45.16, 46.03, 47.18, 47.33, 47.35],
                      'Person': [46.08, 50.51, 56.30, 60.61, 64.21]}
        },
        "x_labels": ['1', '10', '30', '50', '70'],
        "x_label_name": 'Bus passenger occupancy /person',
        "output": 'Sensitivity_to_Occupancy.png'
    },
    "headway": {
        "hypo_peak": {
            'MAPPO-M': {
                'Bus': [108.43, 108.59, 110.23, 109.29, 108.73],
                'Car': [82.01, 66.69, 62.45, 61.31, 60.03],
                'Person': [86.61, 70.04, 64.44, 62.66, 60.72]
            },
            'MAPPO': {
                'Bus': [117.93, 109.31, 107.59, 109.22, 109.44],
                'Car': [98.98, 70.55, 64.62, 63.19, 61.47],
                'Person': [102.28, 73.64, 66.41, 64.49, 62.15]
            }
        },
        "hypo_offpeak": {
            'MAPPO-M': {
                'Bus': [82.26, 83.81, 83.87, 83.92, 82.70],
                'Car': [42.54, 40.05, 38.94, 38.76, 38.36],
                'Person': [54.74, 46.84, 42.72, 41.36, 39.67]
            },
            'MAPPO': {
                'Bus': [95.04, 95.68, 96.12, 96.24, 96.17],
                'Car': [42.56, 40.37, 39.62, 39.30, 39.43],
                'Person': [58.72, 48.95, 44.37, 42.59, 41.11]
            }
        },
        "real_peak": {
            'MAPPO-M': {'Bus': [104.34, 102.48, 99.75, 100.48, 101.18],
                         'Car': [74.70, 62.86, 60.03, 58.70, 58.13],
                         'Person': [81.98, 67.56, 62.53, 60.49, 59.08]},
            'MAPPO': {'Bus': [114.70, 114.58, 105.74, 108.30, 107.80],
                      'Car': [70.42, 60.00, 57.19, 56.66, 55.81],
                      'Person': [81.28, 66.48, 60.25, 58.88, 56.95]}
        },
        "real_offpeak": {
            'MAPPO-M': {'Bus': [95.99, 95.58, 94.85, 95.63, 94.38],
                         'Car': [52.06, 48.03, 47.03, 46.93, 46.34],
                         'Person': [66.75, 56.22, 51.54, 50.09, 47.95]},
            'MAPPO': {'Bus': [98.54, 100.21, 97.18, 97.52, 97.62],
                      'Car': [52.21, 47.18, 46.06, 45.39, 45.12],
                      'Person': [67.73, 56.30, 50.87, 48.76, 46.88]}
        },
        "x_labels": ['2', '5', '10', '15', '30'],
        "x_label_name": 'Bus arrival headway /min',
        "output": 'Sensitivity_to_Headway.png'
    }
}


def parse_args():
    parser = argparse.ArgumentParser(description="Plot sensitivity analysis for occupancy or headway.")
    # Use --metric to switch between occupancy and headway datasets.
    parser.add_argument(
        "--metric",
        choices=["occupancy", "headway"],
        default="headway",
        help="Metric type to plot (occupancy or headway).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Select configuration based on metric
    config = DATA_CONFIG[args.metric]
    x_labels = config["x_labels"]
    x_label_name = config["x_label_name"]
    output_file = config["output"]
    
    # Chart names for 2x2 layout
    chart_names = ['(a) Hypothetical Peak', '(b) Hypothetical Off-Peak',
                   '(c) Real-world Peak', '(d) Real-world Off-Peak']
    
    # Data for each subplot
    data_keys = ['hypo_peak', 'hypo_offpeak', 'real_peak', 'real_offpeak']
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    line_styles = ['-', '--']  # Solid line for MAPPO-M, dashed line for MAPPO
    line_colors = {'Bus': 'blue', 'Car': 'orange', 'Person': 'green'}
    
    for idx, ax in enumerate(axes.flat):
        data_key = data_keys[idx]
        data = config[data_key]
        df = pd.DataFrame(data)
        
        for j, controller in enumerate(df.columns):
            for vehicle in df.index:
                values = df.loc[vehicle, controller]
                line_style = line_styles[j]
                line_color = line_colors[vehicle]
                ax.plot(x_labels, values, linestyle=line_style, marker='o', markersize=4,
                       color=line_color, label=f'{vehicle} - {controller}', linewidth=1.2)
        
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels)
        
        # Keep all labels and ticks for all charts
        ax.set_xlabel(x_label_name)
        ax.set_ylabel('Delay /s')
        
        # Inward ticks on all sides, ensure all labels/ticks are visible
        ax.tick_params(axis='both', which='both', direction='in', 
                      labelleft=True, labelbottom=True, labelright=False, labeltop=False)
        
        # Legend without frame
        ax.legend(frameon=False, loc='best', fontsize=8)
        
        # Panel caption below each subplot
        ax.text(0.5, -0.12, chart_names[idx],
               transform=ax.transAxes, ha='center', va='top', fontsize=12)
    
    fig.tight_layout()
    # Save the figure
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()






