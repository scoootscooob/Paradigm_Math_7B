import matplotlib.pyplot as plt
import numpy as np

# Example data - replace with your actual results
problem_types = ['Arithmetic', 'Algebra', 'Calculus', 'Probability', 'Geometry']
original_accuracy = [0.92, 0.85, 0.78, 0.82, 0.75]
quantized_accuracy = [0.90, 0.83, 0.75, 0.80, 0.72]

x = np.arange(len(problem_types))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))
rects1 = ax.bar(x - width/2, original_accuracy, width, label='Original Model')
rects2 = ax.bar(x + width/2, quantized_accuracy, width, label='Quantized Model')

ax.set_ylabel('Accuracy')
ax.set_title('Mathematical Reasoning Accuracy by Problem Type')
ax.set_xticks(x)
ax.set_xticklabels(problem_types)
ax.legend()

ax.set_ylim(0, 1)
for i, v in enumerate(original_accuracy):
    ax.text(i - width/2, v + 0.02, f"{v:.2f}", ha='center')
for i, v in enumerate(quantized_accuracy):
    ax.text(i + width/2, v + 0.02, f"{v:.2f}", ha='center')

plt.tight_layout()
plt.savefig('/home/ubuntu/paradigm_math_readme/images/math_reasoning_accuracy.png', dpi=300)
