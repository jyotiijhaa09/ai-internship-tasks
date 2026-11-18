import numpy as np

# Vectors and dot product
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print("Dot product:", np.dot(a, b))
print("Cross product:", np.cross(a, b))
print("Norm of a:", np.linalg.norm(a))

# Matrices and linear algebra transformations
A = np.array([[2, 0], [0, 3]])
B = np.array([[1, 2], [3, 4]])
print("Matrix multiply:\n", A @ B)
print("Transpose:\n", B.T)
print("Determinant:", np.linalg.det(B))
print("Inverse:\n", np.linalg.inv(B))

# Apply a transformation (rotate a vector 90 degrees)
theta = np.pi / 2
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])
print("Rotated vector:", np.round(R @ np.array([1, 0]), 2))

# Eigenvalues and eigenvectors
vals, vecs = np.linalg.eig(B)
print("Eigenvalues:", vals)

# Multi-dimensional array operations
arr = np.arange(24).reshape(2, 3, 4)
print("Shape:", arr.shape)
print("Sum over axis 0:\n", arr.sum(axis=0))
print("Mean over axis 2:\n", arr.mean(axis=2))
print("Reshaped:", arr.reshape(4, 6).shape)