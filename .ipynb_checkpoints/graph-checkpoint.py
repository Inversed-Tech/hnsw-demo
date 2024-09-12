import matplotlib.pyplot as plt
import numpy as np

# Data
db_sizes = [1000, 2000, 3000]
ef_64 = [99.90, 99.74, 98.43]
ef_128 = [100, 100, 99.91]
ef_256 = [100, 100, 100]

# Create the plot
plt.figure(figsize=(10, 6))

# Plot lines for each EF value
plt.plot(db_sizes, ef_64, marker='^', label='ef64')
plt.plot(db_sizes, ef_128, marker='s', label='ef128')
plt.plot(db_sizes, ef_256, marker='o', label='ef256')

# Customize the plot
plt.title('efSearch vs DB size', fontsize=12)
plt.xlabel('DB Size', fontsize=10)
plt.ylabel('Accuracy (%)', fontsize=10)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.2)

# Set y-axis limits to focus on the relevant range
plt.ylim(98, 100.5)

# Customize x-axis ticks
plt.xticks(db_sizes)

# Add value labels on the points
for i, db_size in enumerate(db_sizes):
    plt.text(db_size, ef_64[i], '', ha='right', va='bottom')
    # plt.text(db_size, ef_128[i], f'{ef_128[i]}%', ha='right', va='bottom')
    plt.text(db_size, ef_128[i], '', ha='right', va='bottom')
    plt.text(db_size, ef_256[i], '', ha='right', va='top')

# Show the plot
plt.tight_layout()
plt.show()