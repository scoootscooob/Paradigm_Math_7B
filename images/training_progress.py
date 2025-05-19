import matplotlib.pyplot as plt
import numpy as np

# Create training progress visualization
epochs = np.arange(1, 11)
train_loss = [2.8, 2.5, 2.2, 1.9, 1.7, 1.5, 1.4, 1.3, 1.2, 1.1]
eval_loss = [2.9, 2.6, 2.3, 2.0, 1.8, 1.7, 1.6, 1.5, 1.4, 1.3]
learning_rate = [2e-5 * (0.98 ** i) for i in range(10)]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# Loss plot
ax1.plot(epochs, train_loss, 'b-o', label='Training Loss')
ax1.plot(epochs, eval_loss, 'r-o', label='Evaluation Loss')
ax1.set_ylabel('Loss')
ax1.set_title('Training and Evaluation Loss Over Time')
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.7)

# Learning rate plot
ax2.plot(epochs, learning_rate, 'g-o')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Learning Rate')
ax2.set_title('Learning Rate Schedule')
ax2.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('/home/ubuntu/paradigm_math_readme/images/training_progress.png', dpi=300)
