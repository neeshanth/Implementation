import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set the style for a professional look
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 12,
    'axes.titlesize': 16,
    'axes.labelsize': 14
})

# Sample data - replace with your actual DL model performance values
# models = ["ResNet50-UNet", "DenseNet169-UNet", "SE-ResNext50-UNet", "EfficientNetB4-UNet", "Ens4B-UNet", "Our model"]
models = ["MixVision Transformer", "Boundary-aware Transformer", "Our model"]
# accuracy = [0.8346, 0.84, 0.843, 0.82, 0.835, 0.855]
accuracy = [0.8346, 0.848, 0.855]

# Create a DataFrame for easier plotting
df = pd.DataFrame({
    'Model': models,
    'Accuracy': accuracy
})

# Create a figure with appropriate size
plt.figure(figsize=(10, 6))

# Create the barplot with custom colors
ax = sns.barplot(
    x='Model', 
    y='Accuracy', 
    data=df,
    palette='Blues_d',  # Professional color palette
    width=0.6,          # Adjust bar width
    edgecolor='black',  # Add black edges to bars
    linewidth=1         # Edge line width
)

# Add value labels on top of each bar
for i, v in enumerate(accuracy):
    ax.text(i, v + 0.01, f'{v:.2f}', ha='center', fontsize=12)

# Customize the plot
plt.title('Performance Comparison with other SOTA Hybrid Models', pad=15)
plt.xlabel('Model Architecture')
plt.ylabel('Dice Coefficient')
plt.ylim(0.80, 0.95)  # Adjust y-axis to focus on the relevant range

# Add a subtle grid just on the y-axis
ax.yaxis.grid(True, linestyle='--', alpha=0.7)
ax.set_axisbelow(True)

# Rotate x-axis labels slightly if needed
# plt.xticks(rotation=15)

# Tight layout to ensure everything fits nicely
plt.tight_layout()

# Show the plot
plt.show()

# Alternatively, save the figure
# plt.savefig('dl_model_performance.png', dpi=300, bbox_inches='tight')