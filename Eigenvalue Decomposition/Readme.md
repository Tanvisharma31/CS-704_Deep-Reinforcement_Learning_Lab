Eigenvalue decomposition is a fundamental concept in linear algebra, where a matrix is decomposed into a set of eigenvectors and eigenvalues. For a square matrix \(A\), eigenvalue decomposition is given as:

\[
A = V \Lambda V^{-1}
\]

Where:
- \(A\) is the square matrix,
- \(V\) is a matrix containing the eigenvectors,
- \(\Lambda\) is a diagonal matrix containing the eigenvalues,
- \(V^{-1}\) is the inverse of the matrix \(V\).

### Steps for Eigenvalue Decomposition:
1. Compute the eigenvalues and eigenvectors of a matrix.
2. Decompose the matrix using the eigenvalues and eigenvectors.
3. Visualize the relationship between the original matrix, eigenvalues, and eigenvectors.

### Step-by-Step Implementation:

```python
import numpy as np
import matplotlib.pyplot as plt

# Define a simple 2x2 matrix
A = np.array([[4, 1], [2, 3]])

# Perform eigenvalue decomposition
eigenvalues, eigenvectors = np.linalg.eig(A)

# Print the eigenvalues and eigenvectors
print("Matrix A:")
print(A)
print("\nEigenvalues:")
print(eigenvalues)
print("\nEigenvectors:")
print(eigenvectors)

# Reconstruct the original matrix using the eigenvalue decomposition
V = eigenvectors
Lambda = np.diag(eigenvalues)
V_inv = np.linalg.inv(V)
A_reconstructed = np.dot(V, np.dot(Lambda, V_inv))

print("\nReconstructed Matrix A (from eigenvalue decomposition):")
print(A_reconstructed)

# Visualization of eigenvectors
plt.figure(figsize=(6, 6))
origin = [0, 0]  # origin point

# Plot the eigenvectors
plt.quiver(*origin, eigenvectors[0, 0], eigenvectors[1, 0], color='r', scale=5, label='Eigenvector 1')
plt.quiver(*origin, eigenvectors[0, 1], eigenvectors[1, 1], color='b', scale=5, label='Eigenvector 2')

# Plot the transformed vectors by applying the matrix A to the unit vectors
unit_vectors = np.array([[1, 0], [0, 1]])
transformed_vectors = np.dot(A, unit_vectors.T).T

plt.quiver(*origin, transformed_vectors[0, 0], transformed_vectors[0, 1], color='g', scale=5, label='Transformed Vector 1')
plt.quiver(*origin, transformed_vectors[1, 0], transformed_vectors[1, 1], color='purple', scale=5, label='Transformed Vector 2')

plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.grid(True)
plt.legend()
plt.title('Eigenvectors and Transformed Vectors')
plt.show()
```

### Explanation:
1. **Matrix \(A\)**: We define a simple 2x2 matrix `A`.
2. **Eigenvalue Decomposition**: We compute the eigenvalues and eigenvectors of \(A\) using NumPy's `np.linalg.eig()` function.
3. **Reconstruction**: We reconstruct the original matrix \(A\) using the decomposition \(A = V \Lambda V^{-1}\), confirming that eigenvalue decomposition works.
4. **Visualization**: 
   - We plot the eigenvectors of the matrix, which indicate the directions along which the matrix transformation stretches or compresses.
   - We also plot the transformed vectors by applying the matrix \(A\) to the unit vectors \([1, 0]\) and \([0, 1]\) to show how the matrix changes the space.

### Output:
- The eigenvalues represent how much the matrix stretches along the eigenvector directions.
- The eigenvectors represent the directions in which the matrix transformation acts.

### Visualization:
- The red and blue arrows represent the eigenvectors.
- The green and purple arrows show the transformed unit vectors after applying matrix \(A\).
- The eigenvectors are important because the matrix scales the space in the directions of these vectors by the corresponding eigenvalues.

You can modify the matrix \(A\) and observe how the eigenvalues and eigenvectors change.