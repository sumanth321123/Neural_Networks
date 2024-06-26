from minisom import MiniSom
import matplotlib.pyplot as plt
import numpy as np

# Generate random data
data = np.random.rand(100, 2)
target = np.random.choice([0, 1, 2], 100)

# Define SOM grid size
grid_x, grid_y = 10, 10

# Initialize and train the SOM
som = MiniSom(grid_x, grid_y, len(data[0]), sigma=0.3, learning_rate=0.5)
som.random_weights_init(data)
som.train_random(data, 100)

# Plot the SOM distance map
plt.figure(figsize=(10, 10))
plt.pcolor(som.distance_map().T, cmap='bone_r', alpha=0.6)
plt.colorbar()

# Define markers and colors for the target classes
markers = ['o', 's', 'D']
colors = ['r', 'g', 'b']

# Plot data points on the SOM
for i, x in enumerate(data):
    w = som.winner(x)
    plt.plot(w[0] + 0.5, w[1] + 0.5, markers[target[i]],
             markerfacecolor='None',
             markeredgecolor=colors[target[i]], markersize=10,
             markeredgewidth=2)

plt.title('Self-Organizing Map')
plt.show()
