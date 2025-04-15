import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Set the style for a professional look
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 12,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'legend.fontsize': 12,
    'legend.title_fontsize': 13
})

# Load your data - replace with your actual file path
# Example: df = pd.read_csv('your_data.csv')
# For this example, I'll create sample data
df = pd.read_csv("C:/VIT/sem_8/Project - II/Report/Novel_Algo_1_progress.csv")

# Create a figure with appropriate size
plt.figure(figsize=(12, 7))

# Create the lineplot with two columns
# Replace 'x_column', 'y_column1', and 'y_column2' with your actual column names
lineplot = sns.lineplot(
    data=df,
    x='epoch',
    y='train_loss',
    label='Training Loss',  # Change this to a meaningful label
    color='#1f77b4',   # Professional blue color
    linewidth=2.5,
    marker='o',
    markersize=8,
    markerfacecolor='white',
    markeredgewidth=1.5,
    markeredgecolor='#1f77b4'
)

# Add the second line
sns.lineplot(
    data=df,
    x='epoch',
    y='val_loss',
    label='Validation Loss',  # Change this to a meaningful label
    color='#ff7f0e',   # Professional orange color
    linewidth=2.5,
    marker='s',        # Square marker to differentiate
    markersize=8,
    markerfacecolor='white',
    markeredgewidth=1.5,
    markeredgecolor='#ff7f0e'
)

# Customize the plot
plt.title('Phase - 1: Training Loss vs Validation Loss', pad=15)
plt.xlabel('Epoch')  # Change to your actual x-axis label
plt.ylabel('Loss')  # Change to your actual y-axis label

# Add a subtle grid
plt.grid(True, linestyle='--', alpha=0.7)
plt.gca().set_axisbelow(True)

# Customize the legend
legend = plt.legend(
    loc='best',
    frameon=True,
    framealpha=0.95,
    edgecolor='lightgray'
)
legend.get_frame().set_linewidth(1)

# Add annotations if needed (optional)
# plt.annotate('Key point', xy=(x_value, y_value), xytext=(x_value+offset, y_value+offset),
#             arrowprops=dict(arrowstyle='->', color='gray'), fontsize=11)

# Tight layout to ensure everything fits nicely
plt.tight_layout()

# Show the plot
plt.show()

# Alternatively, save the figure
# plt.savefig('comparison_plot.png', dpi=300, bbox_inches='tight')