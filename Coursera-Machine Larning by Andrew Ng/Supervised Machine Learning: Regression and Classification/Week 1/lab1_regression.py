import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x = np.array([1.0, 2.0])       # size in 1000 ft²
y = np.array([300.0, 500.0])   # price in $1000s

def predict(x, w, b):
    """Linear model f_{w,b}(x) = w*x + b (vectorized)."""
    return w * x + b

# Cost function J(w,b)
def compute_cost(x, y, w, b):
    m = x.shape[0]
    resiudals = w * x + b - y # vector of errors
    return resiudals @ resiudals / (2.0 * m)

# Check: for this sata set, the line y=200*x + 100 has zero cost
print(compute_cost(x, y, 200, 100)) # should be 0.0

# How cost changes with w, fixed b=100
b=100.0
w_vals = np.linspace(-100, 400, 501)
J_vals = np.array([compute_cost(x, y, w, b) for w in w_vals])

w_best = w_vals[J_vals.argmin()]

plt.figure()
plt.plot(w_vals, J_vals)
plt.axvline(w_best, linestyle='--')
plt.xlabel('w')
plt.ylabel('J(w, b=100)')
plt.title('Cost function J(w,b) vs w, for b=100')
plt.show()

print("Best w (for b=100) is", w_best) # should be 200.0

# We noticed the curve is bowl-shaped

# Visualiazation of J(w,b) as 3D surface
x_3d=np.array([1.0, 1.7, 2.0, 2.5, 3.0, 3.2])
y_3d = np.array([250, 300, 480, 430, 630, 730])

w_vals_3d = np.linspace(150, 250, 400)
b_vals_3d = np.linspace(-10, 10, 400)
J_vals_3d = np.zeros((len(w_vals_3d), len(b_vals_3d)))

for i, w in enumerate(w_vals_3d):
    for j, b in enumerate(b_vals_3d):
        J_vals_3d[i, j] = compute_cost(x_3d, y_3d, w, b)

# Find minimum cost parameters
min_idx = np.unravel_index(J_vals_3d.argmin(), J_vals_3d.shape)
w_best_3d = w_vals_3d[min_idx[0]]
b_best_3d = b_vals_3d[min_idx[1]]

print("Best parameters: w =", w_best_3d, ", b =", b_best_3d)

#3D Surface plot
fig=plt.figure(figsize=(6,5))
ax=fig.add_subplot(111, projection='3d')
W, B = np.meshgrid(w_vals_3d, b_vals_3d)
ax.plot_surface(W, B, J_vals_3d.T, cmap='viridis')
ax.set_xlabel('w')
ax.set_ylabel('b')
ax.set_zlabel('J(w,b)')
ax.set_title('3D View of Cost Function')
plt.show()

# Contour plot
plt.figure(figsize=(6,5))
plt.contour(W, B, J_vals_3d.T, levels=30, cmap='viridis')
plt.xlabel("w")
plt.ylabel("b")
plt.title("Contour (Top-Down) View of Cost Function")
plt.scatter(w_best_3d, b_best_3d, color='red', marker='x', label="Minimum")
plt.legend()
plt.show()

plt.close('all')
