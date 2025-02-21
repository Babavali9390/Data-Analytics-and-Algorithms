# Naive Bayes Classifier Implementation

Naive Bayes is a probabilistic classifier based on applying Bayes' theorem with strong (naive) independence assumptions between features. This implementation demonstrates the practical application of Naive Bayes classification.

## Features
- Multiple Naive Bayes variants:
  - Gaussian Naive Bayes
  - Multinomial Naive Bayes
  - Bernoulli Naive Bayes
- Laplace smoothing
- Log probability calculations for numerical stability

## Use Cases
- Text Classification
- Spam Detection
- Sentiment Analysis
- Document Categorization
- Medical Diagnosis

## How It Works
1. Calculate prior probabilities for each class
2. Calculate likelihood of features given each class
3. Use Bayes' theorem to calculate posterior probabilities
4. Choose class with highest posterior probability

## Requirements
- Python 3.x
- NumPy
- Pandas
- Scikit-learn (for comparison)

## Usage Example
```python
from naive_bayes import NaiveBayes

# Initialize the classifier
nb = NaiveBayes(type='gaussian')

# Train the model
nb.fit(X_train, y_train)

# Make predictions
predictions = nb.predict(X_test)
```

## Performance Considerations
- Training Time Complexity: O(n*d)
  - n: number of training samples
  - d: number of features
- Prediction Time Complexity: O(c*d)
  - c: number of classes
  - d: number of features

## Advantages
- Fast training and prediction
- Works well with high-dimensional data
- Good for small training sets
- Handles missing values well

## Limitations
- Assumes feature independence
- Sensitive to feature correlation
- Can be outperformed by more sophisticated models

## Best Practices
1. Feature selection to remove correlated features
2. Use appropriate variant for your data type
3. Apply proper text preprocessing for text classification
4. Consider feature scaling for numerical features
