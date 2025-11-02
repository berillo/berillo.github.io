# Vector Creation
import numpy as np

# Allocating and filling by shape
a = np.zeros(4);                
print(f"np.zeros(4) : a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
a = np.zeros((4,));             
print(f"np.zeros(4,) : a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
a = np.random.random_sample(4); 
print(f"np.random.random_sample(4): a = {a}, a shape = {a.shape}, a data type = {a.dtype}")

# Not shape tuple
a = np.arange(4.);              
print(f"np.arange(4.): a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
a = np.random.rand(4);          
print(f"np.random.rand(4): a = {a}, a shape = {a.shape}, a data type = {a.dtype}")

# Manually specifying values
a = np.array([5,4,3,2]);  
print(f"np.array([5,4,3,2]):  a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
a = np.array([5.,4,3,2]); 
print(f"np.array([5.,4,3,2]): a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
# All examples create a 1-D array of shape (4,)

# Operations on Vectors
a = np.arange(10)
print(a)

# element
print(f"a[2].shape: {a[2].shape if hasattr(a[2], 'shape') else ()} a[2]  = {a[2]}, Accessing an element returns a scalar")

# last element
print(f"a[-1] = {a[-1]}")

# out-of-range -> error
try:
    c = a[10]
except Exception as e:
    print("Error: Accessing out-of-range index")
    print(e)

# Slicing

a = np.arange(10)
print(f"a         = {a}")

# five consecutive elements
c = a[2:7:1];     
print("a[2:7:1] = ", c)

# three elements, step 2
c = a[2:7:2];    
print("a[2:7:2] = ", c)

# index 3 and above
c = a[3:];        
print("a[3:]= ", c)

# below index 3
c = a[:3];        
print("a[:3]= ", c)

# all elements
c = a[:];         
print("a[:]= ", c)

# Single-vector operations
a = np.array([1,2,3,4])
print(f"a: {a}")

b = -a 
print(f"b = -a: {b}")

b = np.sum(a) 
print(f"b = np.sum(a): {b}")

b = np.mean(a)
print(f"b = np.mean(a): {b}")

b = a**2
print(f"b = a**2: {b}")

# Vector–Vector element-wise ops
#   Binary operators work element wise
a = np.array([1,2,3,4])
b = np.array([-1,-2, 3, 4])
print(f"Binary operators work element wise: {a + b}")

# mismatched shapes
c = np.array([1, 2])
try:
    d = a + c
except Exception as e:
    print("Error: Mismatched shapes in vector-vector operations")
    print(e)

# Scalar–Vector operations
a = np.array([1,2,3,4])
b = 5 * a
print(f"Scalar–Vector operations: b = 5 * a = {b}")

# Vector–Vector dot product
def my_dot(a, b): 
    x = 0
    for i in range(a.shape[0]):
        x = x + a[i] * b[i]
    return x

a = np.array([1,2,3,4])
b = np.array([-1, 4, 3, 2])
print(f"my_dot(a, b) = {my_dot(a, b)}")

# Using NumPy built-in dot function
a = np.array([1, 2, 3, 4])
b = np.array([-1, 4, 3, 2])
c = np.dot(a, b)
print(f"NumPy 1-D np.dot(a, b) = {c}, np.dot(a, b).shape = {c.shape} ") 
c = np.dot(b, a)
print(f"NumPy 1-D np.dot(b, a) = {c}, np.dot(a, b).shape = {c.shape} ")

# The Need for Speed: vectorization vs loop
#   Vectorization is much faster because NumPy uses optimized, parallel native code
import time

np.random.seed(1)
a = np.random.rand(10_000_000)  # large arrays
b = np.random.rand(10_000_000)

tic = time.time()
c = np.dot(a, b)
toc = time.time()
print(f"np.dot(a, b) =  {c:.4f}")
print(f"Vectorized version duration: {1000*(toc-tic):.4f} ms ")

tic = time.time()
c = my_dot(a,b)
toc = time.time()
print(f"my_dot(a, b) =  {c:.4f}")
print(f"loop version duration: {1000*(toc-tic):.4f} ms ")

del(a)
del(b)
# Expect a large speedup from the vectorized (np.dot) version.

# Vector–Vector ops in Course 1
#   Training data 𝑋 will be shape (m, n) (m examples, n features).
#   Parameter vector w is (n,).
#   Often we extract a single example: X[i] → shape (n,), and do vector-vector ops like a dot product.

X = np.array([[1],[2],[3],[4]])
w = np.array([2])
c = np.dot(X[1], w)

print(f"X[1] has shape {X[1].shape}")
print(f"w has shape {w.shape}")
print(f"c has shape {c.shape}")

# Matrices

# A matrix is a 2-D array (all elements same dtype), denoted with bold uppercase (e.g., X).
#   We use m rows and n columns.
#   Matrix elements use two indices: [row, column].
#   In code: rows/cols are 0..m-1 and 0..n-1.
# Generic matrix notation: first index is row, second is column.

# NumPy Arrays (2-D)
# Matrix Creation
a = np.zeros((1, 5))                                       
print(f"a shape = {a.shape}, a = {a}")

a = np.zeros((2, 1))                                                                   
print(f"a shape = {a.shape}, a = {a}")

a = np.random.random_sample((1, 1))  
print(f"a shape = {a.shape}, a = {a}")

# Manual construction:
a = np.array([[5], [4], [3]]);   print(f" a shape = {a.shape}, np.array: a = {a}")
a = np.array([[5],   # One can also
              [4],   # separate values
              [3]]); # into separate rows
print(f" a shape = {a.shape}, np.array: a = {a}")

# Matrix Operations
# Indexing

a = np.arange(6).reshape(-1, 2)   # convenient way to create matrices
print(f"a.shape: {a.shape}, \na= {a}")

# element
print(f"\na[2,0].shape:   {np.array(a[2,0]).shape}, a[2,0] = {a[2,0]},     type(a[2,0]) = {type(a[2,0])} Accessing an element returns a scalar\n")

# row
print(f"a[2].shape:   {a[2].shape}, a[2]   = {a[2]}, type(a[2])   = {type(a[2])}")

# Slicing
#  Use the same start:stop:step idea, now with two dimensions.
a = np.arange(20).reshape(-1, 10)
print(f"a = \n{a}")

# 5 consecutive elements from row 0
print("a[0, 2:7:1] = ", a[0, 2:7:1], ",  a[0, 2:7:1].shape =", a[0, 2:7:1].shape, "a 1-D array")

# same columns across all rows
print("a[:, 2:7:1] = \n", a[:, 2:7:1], ",  a[:, 2:7:1].shape =", a[:, 2:7:1].shape, "a 2-D array")

# all elements
print("a[:,:] = \n", a[:,:], ",  a[:,:].shape =", a[:,:].shape)

# all elements of a single row (very common)
print("a[1,:] = ", a[1,:], ",  a[1,:].shape =", a[1,:].shape, "a 1-D array")
# same as:
print("a[1]   = ", a[1],   ",  a[1].shape   =", a[1].shape, "a 1-D array")