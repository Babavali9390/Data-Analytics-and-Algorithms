# K-Means Clustering Implementation

K-Means is an unsupervised machine learning algorithm used to identify and cluster similar data points into groups. This implementation demonstrates the practical application of K-Means clustering.

## Features
- Customizable number of clusters (k)
- Iterative refinement technique
- Centroid-based clustering
- Euclidean distance calculation

## Use Cases
- Customer Segmentation
- Image Compression
- Pattern Recognition
- Document Classification
- Feature Learning

## How It Works
1. Initialize k centroids randomly
2. Assign each data point to the nearest centroid
3. Update centroids by calculating mean of all points in the cluster
4. Repeat steps 2-3 until convergence

## Requirements
- Python 3.x
- NumPy
- Matplotlib (for visualization)
- Scikit-learn (for comparison and evaluation)

## Usage Example
```python
from kmeans import KMeans

# Initialize the model
kmeans = KMeans(n_clusters=3)

# Fit the model
kmeans.fit(data)

# Get cluster assignments
labels = kmeans.predict(data)
```

## Performance Considerations
- Time Complexity: O(n*k*i)
  - n: number of points
  - k: number of clusters
  - i: number of iterations

## Limitations
- Requires pre-specified number of clusters
- Sensitive to initial centroid positions
- May converge to local optima
- Assumes spherical clusters

## References
- MacQueen, J. (1967). Some methods for classification and analysis of multivariate observations.
- Lloyd, S. P. (1982). Least squares quantization in PCM.
