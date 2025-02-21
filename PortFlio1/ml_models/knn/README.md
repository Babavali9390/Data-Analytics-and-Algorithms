# K-Nearest Neighbors (KNN) Implementation

KNN is a simple yet powerful supervised machine learning algorithm that can be used for both classification and regression tasks. This implementation focuses on the classification aspect of KNN.

## Features
- Configurable number of neighbors (k)
- Multiple distance metrics support
- Weighted voting option
- Cross-validation support

## Use Cases
- Classification Tasks
- Recommendation Systems
- Pattern Recognition
- Missing Data Imputation
- Anomaly Detection

## How It Works
1. Store all training data points
2. For each new point:
   - Calculate distance to all training points
   - Find k nearest neighbors
   - Take majority vote (classification) or average (regression)
   - Assign class/value to new point

## Requirements
- Python 3.x
- NumPy
- Pandas
- Scikit-learn (for comparison and evaluation)

## Usage Example
```python
from knn import KNNClassifier

# Initialize the classifier
knn = KNNClassifier(k=3, metric='euclidean')

# Train the model
knn.fit(X_train, y_train)

# Make predictions
predictions = knn.predict(X_test)
```

## Performance Considerations
- Time Complexity: O(n*d) for each prediction
  - n: number of training samples
  - d: number of features

## Advantages
- No training phase
- Simple to implement
- Naturally handles multi-class cases
- Can be used for both classification and regression

## Limitations
- Computationally expensive during prediction
- Requires feature scaling
- Sensitive to irrelevant features
- Memory-intensive (stores all training data)

## Tips for Better Performance
1. Choose k wisely (typically sqrt(n))
2. Scale features appropriately
3. Use dimensional reduction if needed
4. Consider using weighted voting for better accuracy
