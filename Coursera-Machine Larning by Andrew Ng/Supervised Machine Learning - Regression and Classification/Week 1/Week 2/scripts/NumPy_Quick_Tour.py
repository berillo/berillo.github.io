# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
# ---

# %%
# ------------------------------------------------------------
# NumPy Quick Tour: Vectors, Matrices, and Vectorization
# Author: Beril Ipek Erdem
#
# This notebook provides an educational overview of NumPy essentials used
# in scientific computing and machine learning. It demonstrates how to create,
# index, and slice arrays; perform element-wise, scalar, and vector operations;
# and highlights the efficiency of vectorization over loops.
#
# The notebook also explores 2-D matrices, broadcasting, and reshape behavior.
# All examples, explanations, and code are original and written for clarity.
#
# License: MIT — free to use, modify, and share for educational purposes
# with proper attribution to the author.
# ------------------------------------------------------------

# %%
import numpy as np
import time

# %%
# ## 1) Vectors: creation

# %%
print("\n=== 1) Vectors: creation ===")
v = np.zeros(5)
print(f"zeros -> v={v!r}, shape={v.shape}, dtype={v.dtype}")

v = np.zeros((5,))   # explicit 1-D shape
print(f"zeros((5,)) -> shape={v.shape}")

v = np.random.random_sample(5)
print(f"random_sample -> v={v}, shape={v.shape}")

# Routines that don't take a shape tuple
v = np.arange(5.0)           # 0.0..4.0
print(f"arange -> v={v}, shape={v.shape}")

v = np.random.rand(5)        # uniform [0,1)
print(f"rand -> v={v}, shape={v.shape}")

# Manual values (note dtype difference)
v = np.array([9, 7, 5, 3])
print(f"array ints -> {v}, dtype={v.dtype}")
v = np.array([9., 7, 5, 3])
print(f"array floats -> {v}, dtype={v.dtype}")

# %%
# ## 2) Indexing & slicing on vectors

# %%
print("\n=== 2) Indexing & slicing (1-D) ===")
a = np.arange(12)     # [0..11]
print("a =", a)

# single element
print("a[3] ->", a[3], "(scalar)")

# last element (negative index)
print("a[-1] ->", a[-1])

# out of range -> exception
try:
    _ = a[99]
except Exception as e:
    print("Out-of-range example ->", type(e).__name__, "-", e)

# slices (start:stop:step)
print("a[4:10]   ->", a[4:10])
print("a[2:11:3] ->", a[2:11:3])
print("a[:5]     ->", a[:5])
print("a[7:]     ->", a[7:])
print("a[:]      ->", a[:])   # full copy view

# %%
# ## 3) Single vector ops

# %%
print("\n=== 3) Single vector ops ===")
b = np.array([1, 2, 3, 4])
print("b:", b)
print("-b             ->", -b)
print("sum(b)         ->", np.sum(b))
print("mean(b)        ->", np.mean(b))
print("b ** 2         ->", b ** 2)

# %%
# ## 4) Element-wise ops & broadcasting

# %%
print("\n=== 4) Element-wise ops & broadcasting ===")
x = np.array([2, 4, 6, 8])
y = np.array([1, -1, 3, -3])
print("x + y         ->", x + y)

# mismatched shapes normally fail:
short = np.array([10, 20])
try:
    _ = x + short
except ValueError as e:
    print("Shape mismatch example ->", e)

# broadcasting in compatible cases:
col = np.arange(3).reshape(3, 1)    # shape (3,1)
row = np.array([10, 20, 30])        # shape (3,)
print("Broadcasting demo:\n", col + row.reshape(1, 3))  # (3,3)

# %%
# ## 5) Scalar–vector ops

# %%
print("\n=== 5) Scalar–vector ops ===")
print("5 * x ->", 5 * x)

# %%
# ## 6) Dot product: loop vs `np.dot`  

# %%
print("\n=== 6) Dot product ===")

def dot_loop(u: np.ndarray, v: np.ndarray) -> float:
    """Pure-Python loop dot product; assumes same length."""
    acc = 0.0
    for i in range(u.shape[0]):
        acc += u[i] * v[i]
    return acc

u = np.array([1, 3, -2, 5], dtype=float)
v = np.array([4, -1, 0.5, 2], dtype=float)

print("dot_loop(u,v) ->", dot_loop(u, v))
print("np.dot(u, v)  ->", np.dot(u, v))

# %%
# ## 7) Why vectorization is fast (timing demo)

# %%
print("\n=== 7) Vectorization timing ===")
np.random.seed(42)
N = 5_000_000  # keep reasonable for demos
u_big = np.random.rand(N)
v_big = np.random.rand(N)

t0 = time.time()
dp_vec = np.dot(u_big, v_big)
t1 = time.time()
print(f"np.dot: {dp_vec:.4f}, time = {(t1 - t0)*1e3:.2f} ms")

t0 = time.time()
dp_loop = dot_loop(u_big, v_big)
t1 = time.time()
print(f"loop : {dp_loop:.4f}, time = {(t1 - t0)*1e3:.2f} ms")

del u_big, v_big  # free memory

# %%
# ## 8) Matrices: creation & indexing

# %%
print("\n=== 8) Matrices: creation & indexing ===")
M = np.zeros((2, 3))
print("zeros((2,3)) ->\n", M, "\nshape:", M.shape)

M = np.random.random_sample((3, 1))
print("random_sample((3,1)) ->\n", M, "\nshape:", M.shape)

# manual construction
M = np.array([[7, 8, 9],
              [1, 2, 3],
              [4, 5, 6]])
print("manual M ->\n", M)

# element (row, col)
print("M[1, 2] ->", M[1, 2])

# row view (returns 1-D)
print("M[0]    ->", M[0], ", shape:", M[0].shape)

# %%
# ## 9) Matrix slicing

# %%
print("\n=== 9) Matrix slicing ===")
A = np.arange(24).reshape(4, 6)
print("A =\n", A)

print("A[1, 2:5]   ->", A[1, 2:5])           # 1-D slice of a row
print("A[:, 1:4]   ->\n", A[:, 1:4])         # submatrix
print("A[::2, ::3] ->\n", A[::2, ::3])       # stride in both dims
print("A[3, :]     ->", A[3, :], "(row)")
print("A[:, 0]     ->", A[:, 0], "(column)")

# %%
# ## 10) Reshape gotchas

# %%
print("\n=== 10) Reshape gotchas ===")
w = np.arange(6)        # shape (6,)
print("w:", w, "shape:", w.shape)

W2 = w.reshape(2, 3)    # (2,3)
print("W2:\n", W2, "shape:", W2.shape)

# Extract row returns 1-D vector
row0 = W2[0]
print("W2[0] ->", row0, "shape:", row0.shape)

# If you need a *row as 2-D*, keep dims:
row0_2d = W2[0:1, :]
print("W2[0:1,:] ->", row0_2d, "shape:", row0_2d.shape)

# And a column as 2-D:
col1_2d = W2[:, 1:2]
print("W2[:,1:2] ->\n", col1_2d, "shape:", col1_2d.shape)

print("\nAll demos complete.")
