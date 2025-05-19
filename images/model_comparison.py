import matplotlib.pyplot as plt
import numpy as np

# Example data - replace with your actual results
models = ['Original', '8-bit Quantized', '4-bit Quantized']
perplexity = [6.8, 7.1, 7.5]  # Lower is better
inference_speed = [12, 22, 35]  # Tokens per second, higher is better
model_size = [14, 7, 3.5]  # GB, lower is better

# Create figure with multiple subplots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

# Perplexity (lower is better)
ax1.bar(models, perplexity, color='skyblue')
ax1.set_ylabel('Perplexity (lower is better)')
ax1.set_title('Model Perplexity')
for i, v in enumerate(perplexity):
    ax1.text(i, v + 0.1, f"{v:.1f}", ha='center')

# Inference Speed (higher is better)
ax2.bar(models, inference_speed, color='lightgreen')
ax2.set_ylabel('Tokens per second (higher is better)')
ax2.set_title('Inference Speed')
for i, v in enumerate(inference_speed):
    ax2.text(i, v + 1, f"{v}", ha='center')

# Model Size (lower is better)
ax3.bar(models, model_size, color='salmon')
ax3.set_ylabel('Size in GB (lower is better)')
ax3.set_title('Model Size')
for i, v in enumerate(model_size):
    ax3.text(i, v + 0.3, f"{v:.1f} GB", ha='center')

plt.tight_layout()
plt.savefig('/home/ubuntu/paradigm_math_readme/images/model_comparison.png', dpi=300)
