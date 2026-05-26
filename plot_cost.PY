import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# 1. Generate noisy house data completely flat
np.random.seed(42)
x_data = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
y_perfect = 1.0 + 2.0 * x_data
y_data = y_perfect + np.random.normal(0, 0.6, size=6)
m = 6

# 2. Compute 3D Cost grid using fast vectorized matrix operations (No loops!)
t0_range = np.linspace(-2, 6, 100)
t1_range = np.linspace(-1, 5, 100)
T0, T1 = np.meshgrid(t0_range, t1_range)
Z_cost = np.sum((T0[..., np.newaxis] + T1[..., np.newaxis] * x_data - y_data) ** 2, axis=-1) / (2 * m)

# 3. Find optimal parameter minimums
min_idx = np.unravel_index(np.argmin(Z_cost), Z_cost.shape)
best_t0 = T0[min_idx]
best_t1 = T1[min_idx]
min_cost = Z_cost[min_idx]
fig = plt.figure(figsize=(15, 6))

# Left Plot: 3D Surface
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
surf = ax1.plot_surface(T0, T1, Z_cost, cmap=cm.plasma, edgecolor='none', alpha=0.8)
fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=10, label='Cost J')
ax1.scatter([best_t0], [best_t1], [min_cost], color='red', marker='*', s=200, label='Minimum')
ax1.set_xlabel('Intercept (theta0)')
ax1.set_ylabel('Slope (theta1)')
ax1.set_zlabel('Cost J')
ax1.set_title('3D Surface: Noisy Cost Bowl')

# Right Plot: Data Fit Line
ax2 = fig.add_subplot(1, 2, 2)
ax2.scatter(x_data, y_data, color='darkorange', s=100, label='Noisy Data')
x_line = np.linspace(0, 4, 100)
y_line = best_t0 + best_t1 * x_line
ax2.plot(x_line, y_line, color='blue', lw=2, label='Line of Best Fit')
ax2.set_xlabel('House Size')
ax2.set_ylabel('House Price')
ax2.set_title('Linear Regression Fit')
ax2.grid(True, linestyle='--')
ax2.legend()

plt.tight_layout()
plt.show()