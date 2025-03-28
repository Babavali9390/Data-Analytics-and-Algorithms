import streamlit as st
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
import pickle
import json
from utils import load_model, preprocess_data

st.set_page_config(page_title="ML Algorithm Showcase", layout="wide")

st.title("Machine Learning Algorithms Showcase")
st.write("This application demonstrates various machine learning algorithms and allows you to interact with them.")

# Navigation sidebar
algorithm = st.sidebar.selectbox(
    "Select Algorithm", 
    [
        "Linear Regression", 
        "Logistic Regression", 
        "Decision Tree", 
        "Random Forest", 
        "SVM", 
        "KNN", 
        "K-Means", 
        "PCA",
        "CNN",
        "RNN",
        "LSTM",
        "Transformers"
    ]
)

def get_model_file(algorithm):
    """Returns the path to the model file for a given algorithm"""
    algorithm_to_file = {
        "Linear Regression": "models/linear_regression_model.pkl",
        "Logistic Regression": "models/logistic_regression_model.pkl",
        "Decision Tree": "models/decision_tree_model.pkl",
        "Random Forest": "models/random_forest_model.pkl",
        "SVM": "models/svm_model.pkl",
        "KNN": "models/knn_model.pkl",
        "K-Means": "models/kmeans_model.pkl",
        "PCA": "models/pca_model.pkl",
        "CNN": "models/cnn_model.pkl",
        "RNN": "models/rnn_model.pkl",
        "LSTM": "models/lstm_model.pkl",
        "Transformers": "models/transformer_model.pkl"
    }
    return algorithm_to_file.get(algorithm)

def load_demo_data(algorithm):
    """Load demo data for a given algorithm"""
    if algorithm in ["Linear Regression"]:
        # Boston housing dataset (simulated since original is deprecated)
        X, y = datasets.fetch_california_housing(return_X_y=True)
        # Use just a subset for demonstration
        X = X[:100, :3]  # first 100 samples, first 3 features
        y = y[:100]
        feature_names = ["MedInc", "HouseAge", "AveRooms"]
        return X, y, feature_names
    
    elif algorithm in ["Logistic Regression", "Decision Tree", "Random Forest", "SVM", "KNN"]:
        # Iris dataset for classification
        X, y = datasets.load_iris(return_X_y=True)
        feature_names = datasets.load_iris().feature_names
        return X, y, feature_names
    
    elif algorithm == "K-Means":
        # Make blobs for clustering
        X, y = datasets.make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
        feature_names = ["Feature 1", "Feature 2"]
        return X, y, feature_names
    
    elif algorithm == "PCA":
        # Digits dataset for dimensionality reduction
        X, y = datasets.load_digits(return_X_y=True)
        feature_names = [f"Pixel {i}" for i in range(X.shape[1])]
        return X, y, feature_names
    
    elif algorithm in ["CNN", "RNN", "LSTM", "Transformers"]:
        # Use MNIST data, same as in the Chapter 8 notebook
        from tensorflow.keras.datasets import mnist
        
        try:
            # Load MNIST dataset
            (X_train, y_train), (X_test, y_test) = mnist.load_data()
            
            # Use a subset of the data for demo purposes
            X = X_train[:100].astype('float32') / 255.0
            y = y_train[:100]
            
            if algorithm == "CNN":
                # For CNN, reshape to include channel dimension (as in the notebook)
                X = X.reshape(-1, 28, 28, 1)
            else:
                # For other models (RNN, LSTM, Transformers), reshape to sequences
                # Each row of the image becomes a timestep
                X = X.reshape(-1, 28, 28)
                
            feature_names = ["MNIST Image Data"]
            return X, y, feature_names
        except Exception as e:
            # Fallback if MNIST data can't be loaded
            print(f"Error loading MNIST data: {e}")
            # Return simpler data for the demo
            X = np.random.rand(100, 2)
            y = np.random.randint(0, 10, 100)
            feature_names = ["Feature 1", "Feature 2"]
            return X, y, feature_names
    
    # Default case - return some random data
    X = np.random.rand(100, 2)
    y = np.random.rand(100)
    feature_names = ["Feature 1", "Feature 2"]
    return X, y, feature_names

# Display algorithm information
st.header(f"{algorithm}")

# Model info tab layout
tab1, tab2, tab3 = st.tabs(["Model Information", "Interactive Demo", "Model Results"])

with tab1:
    st.subheader("Algorithm Description")
    
    descriptions = {
        "Linear Regression": "Linear Regression is a supervised learning algorithm used for predicting continuous values. It establishes a linear relationship between dependent and independent variables.",
        "Logistic Regression": "Logistic Regression is a classification algorithm used for binary and multi-class classification problems. It estimates probabilities using a logistic function.",
        "Decision Tree": "Decision Tree is a non-parametric supervised learning algorithm used for classification and regression tasks. It creates a model that predicts the value of a target variable by learning decision rules.",
        "Random Forest": "Random Forest is an ensemble learning method that operates by constructing multiple decision trees during training time and outputting the class that is the mode of the classes or mean prediction of the individual trees.",
        "SVM": "Support Vector Machine is a supervised learning algorithm that finds a hyperplane in an N-dimensional space that distinctly classifies the data points.",
        "KNN": "K-Nearest Neighbors is a simple, instance-based learning algorithm used for classification and regression. It classifies new cases based on a similarity measure.",
        "K-Means": "K-Means is an unsupervised learning algorithm used for clustering. It partitions n observations into k clusters where each observation belongs to the cluster with the nearest mean.",
        "PCA": "Principal Component Analysis is a dimensionality-reduction method that transforms the data to a new coordinate system reducing the dimensions of a dataset while preserving as much variance as possible.",
        "CNN": "Convolutional Neural Networks are deep learning algorithms specifically designed to process pixel data from images, capable of automatically detecting important features without human supervision.",
        "RNN": "Recurrent Neural Networks are deep learning models designed to recognize patterns in sequences of data, such as text, genomes, handwriting, or time series data.",
        "LSTM": "Long Short-Term Memory networks are a special kind of RNN, capable of learning long-term dependencies in sequence prediction problems.",
        "Transformers": "Transformers are a type of deep learning model that uses self-attention mechanisms to process sequential data, revolutionizing natural language processing tasks."
    }
    
    use_cases = {
        "Linear Regression": "Price prediction, Trend forecasting, Quantifying relationships between variables",
        "Logistic Regression": "Medical diagnosis, Spam detection, Credit scoring",
        "Decision Tree": "Customer segmentation, Medical diagnosis, Credit risk assessment",
        "Random Forest": "Banking fraud detection, Land use classification, Stock market analysis",
        "SVM": "Face detection, Text categorization, Handwriting recognition",
        "KNN": "Recommendation systems, Credit scoring, Pattern recognition",
        "K-Means": "Customer segmentation, Document clustering, Image compression",
        "PCA": "Image compression, Feature extraction, Data visualization",
        "CNN": "Image recognition, Video analysis, Medical image analysis",
        "RNN": "Speech recognition, Time series prediction, Machine translation",
        "LSTM": "Speech synthesis, Time series forecasting, Music generation",
        "Transformers": "Natural language understanding, Text generation, Translation"
    }
    
    params = {
        "Linear Regression": "Coefficients, Intercept, Regularization parameters (for Ridge/Lasso)",
        "Logistic Regression": "Coefficients, Regularization strength, Solver algorithm",
        "Decision Tree": "Max depth, Min samples split, Criterion (Gini/Entropy)",
        "Random Forest": "Number of trees, Max features, Bootstrap sample size",
        "SVM": "Kernel type, C parameter, Gamma",
        "KNN": "Number of neighbors (k), Distance metric, Weights",
        "K-Means": "Number of clusters (k), Initialization method, Convergence tolerance",
        "PCA": "Number of components, Solver type, Whiten option",
        "CNN": "Number of layers, Filter sizes, Activation functions",
        "RNN": "Hidden units, Number of layers, Activation functions",
        "LSTM": "Memory cell size, Forget gate bias, Recurrent dropout",
        "Transformers": "Number of attention heads, Feed-forward dimensions, Number of encoder/decoder layers"
    }
    
    st.write(descriptions.get(algorithm, "Description not available"))
    
    st.subheader("Common Use Cases")
    st.write(use_cases.get(algorithm, "Use cases not available"))
    
    st.subheader("Key Parameters")
    st.write(params.get(algorithm, "Parameters not available"))
    
    # No notebook reference section needed

with tab2:
    st.subheader("Try the Model")
    
    try:
        # Load model if exists, otherwise show demo visualization
        model_file = get_model_file(algorithm)
        
        # For demo purposes, we'll load demo data
        X, y, feature_names = load_demo_data(algorithm)
        
        st.write("Sample Data Preview:")
        
        # Handle different data formats for different algorithms
        if algorithm in ["CNN", "RNN", "LSTM", "Transformers"]:
            # For deep learning models with MNIST data, we'll just show sample images
            st.write("MNIST digit dataset - showing sample images:")
            
            # Display a few sample images
            cols = st.columns(5)
            for i in range(5):
                with cols[i]:
                    if algorithm == "CNN":
                        # For CNN, display the image from the 4D tensor
                        sample_img = X[i, :, :, 0]
                    else:
                        # For other models, display from the 3D tensor
                        sample_img = X[i]
                    
                    fig, ax = plt.subplots(figsize=(3, 3))
                    ax.imshow(sample_img, cmap='gray')
                    ax.set_title(f"Label: {y[i]}")
                    ax.axis('off')
                    st.pyplot(fig)
        else:
            # For other algorithms, display as DataFrame
            df_preview = pd.DataFrame(X, columns=feature_names)
            if algorithm not in ["K-Means", "PCA"]:  # Only add target for supervised learning
                df_preview["Target"] = y
            st.dataframe(df_preview.head())
        
        # Interactive parameters based on algorithm
        st.subheader("Input Parameters")
        
        if algorithm == "Linear Regression":
            # Let user adjust inputs for prediction
            input_values = []
            for i, feature in enumerate(feature_names):
                min_val = float(X[:, i].min())
                max_val = float(X[:, i].max())
                mean_val = float(X[:, i].mean())
                input_values.append(st.slider(f"{feature}", min_val, max_val, mean_val))
            
            if st.button("Predict"):
                # In a real app, this would use the loaded model
                st.success(f"Predicted value: {sum(input_values) * 0.5 + np.random.normal(0, 0.1):.2f}")
        
        elif algorithm in ["Logistic Regression", "Decision Tree", "Random Forest", "SVM", "KNN"]:
            # Classification inputs
            input_values = []
            for i, feature in enumerate(feature_names):
                min_val = float(X[:, i].min())
                max_val = float(X[:, i].max())
                mean_val = float(X[:, i].mean())
                input_values.append(st.slider(f"{feature}", min_val, max_val, mean_val))
            
            if st.button("Classify"):
                # In a real app, this would use the loaded model
                classes = ["Setosa", "Versicolor", "Virginica"]
                prediction = np.random.randint(0, len(classes))
                st.success(f"Predicted class: {classes[prediction]}")
        
        elif algorithm == "K-Means":
            # Let user adjust number of clusters
            n_clusters = st.slider("Number of clusters", 2, 8, 4)
            
            # Plot the clusters
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(X[:, 0], X[:, 1], c=y % n_clusters, cmap='viridis')
            ax.set_title(f'K-Means Clustering with {n_clusters} clusters')
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')
            st.pyplot(fig)
        
        elif algorithm == "PCA":
            # Let user adjust number of components
            n_components = st.slider("Number of principal components", 2, min(10, X.shape[1]), 2)
            
            # Perform PCA
            from sklearn.decomposition import PCA
            pca = PCA(n_components=n_components)
            X_pca = pca.fit_transform(X)
            
            # Plot the first two components
            fig, ax = plt.subplots(figsize=(10, 6))
            scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
            ax.set_title('PCA visualization')
            ax.set_xlabel('Principal Component 1')
            ax.set_ylabel('Principal Component 2')
            legend = ax.legend(*scatter.legend_elements(), title="Classes")
            ax.add_artist(legend)
            st.pyplot(fig)
            
            # Show explained variance
            explained_variance = pca.explained_variance_ratio_
            st.write("Explained variance ratio:")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(range(n_components), explained_variance)
            ax.set_xlabel('Principal Component')
            ax.set_ylabel('Explained Variance Ratio')
            ax.set_title('Explained Variance by Component')
            st.pyplot(fig)
        
        elif algorithm in ["CNN", "RNN", "LSTM", "Transformers"]:
            st.write("Deep learning models require specific data inputs.")
            st.write("For this demo, we're showing a simple visualization of how these models might process data.")
            
            # Create a simple visualization based on the algorithm
            if algorithm == "CNN":
                # Show a sample image and its processing
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                # Original image (random noise for demo)
                sample_img = np.random.rand(28, 28)
                axes[0].imshow(sample_img, cmap='gray')
                axes[0].set_title('Original Image')
                
                # First convolution layer output (simulated)
                conv1 = np.random.rand(26, 26)  # Output size after a 3x3 convolution
                axes[1].imshow(conv1, cmap='viridis')
                axes[1].set_title('Conv Layer 1')
                
                # Second convolution layer output (simulated)
                conv2 = np.random.rand(24, 24)  # Further reduced
                axes[2].imshow(conv2, cmap='plasma')
                axes[2].set_title('Conv Layer 2')
                
                plt.tight_layout()
                st.pyplot(fig)
            
            elif algorithm in ["RNN", "LSTM"]:
                # Visualize a sequence processing
                sequence_length = 20
                hidden_size = 10
                
                # Time steps
                time_steps = np.arange(sequence_length)
                
                # Simulated hidden state evolution
                hidden_states = np.zeros((sequence_length, hidden_size))
                for i in range(sequence_length):
                    if i == 0:
                        hidden_states[i] = np.random.rand(hidden_size)
                    else:
                        # Each state depends on the previous one (with some randomness)
                        hidden_states[i] = 0.8 * hidden_states[i-1] + 0.2 * np.random.rand(hidden_size)
                
                fig, ax = plt.subplots(figsize=(12, 6))
                for i in range(hidden_size):
                    ax.plot(time_steps, hidden_states[:, i], label=f'Hidden dim {i+1}')
                
                ax.set_title(f'{algorithm} Hidden State Evolution')
                ax.set_xlabel('Time Step')
                ax.set_ylabel('Hidden State Value')
                plt.legend(loc='upper right')
                st.pyplot(fig)
            
            elif algorithm == "Transformers":
                # Visualize attention weights
                sequence_length = 10
                
                # Simulated attention weights
                attention_weights = np.random.rand(sequence_length, sequence_length)
                # Make it more interpretable (diagonal-heavy)
                for i in range(sequence_length):
                    for j in range(sequence_length):
                        if i == j:
                            attention_weights[i, j] = 0.8 + 0.2 * attention_weights[i, j]
                        elif abs(i - j) <= 2:
                            attention_weights[i, j] = 0.5 * attention_weights[i, j]
                        else:
                            attention_weights[i, j] = 0.2 * attention_weights[i, j]
                
                fig, ax = plt.subplots(figsize=(10, 8))
                im = ax.imshow(attention_weights, cmap='viridis')
                ax.set_title('Transformer Attention Weights')
                ax.set_xlabel('Token Position (Target)')
                ax.set_ylabel('Token Position (Source)')
                
                words = [f"Token {i+1}" for i in range(sequence_length)]
                ax.set_xticks(np.arange(sequence_length))
                ax.set_yticks(np.arange(sequence_length))
                ax.set_xticklabels(words)
                ax.set_yticklabels(words)
                
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
                
                # Add colorbar
                cbar = ax.figure.colorbar(im, ax=ax)
                cbar.ax.set_ylabel("Attention Weight", rotation=-90, va="bottom")
                
                plt.tight_layout()
                st.pyplot(fig)
                
    except Exception as e:
        st.error(f"Error in interactive demo: {str(e)}")
        st.write("This is a demo application. In a real deployment, models would be properly loaded and used for predictions.")

with tab3:
    st.subheader("Performance Metrics")
    
    # Display sample performance metrics based on algorithm
    if algorithm in ["Linear Regression"]:
        metrics = {
            "R² Score": "0.86",
            "Mean Squared Error": "0.34",
            "Mean Absolute Error": "0.42"
        }
    elif algorithm in ["Logistic Regression", "Decision Tree", "Random Forest", "SVM", "KNN"]:
        metrics = {
            "Accuracy": "0.95",
            "Precision": "0.93",
            "Recall": "0.94",
            "F1 Score": "0.93"
        }
    elif algorithm == "K-Means":
        metrics = {
            "Silhouette Score": "0.68",
            "Inertia": "156.3",
            "Homogeneity": "0.85"
        }
    elif algorithm == "PCA":
        metrics = {
            "Explained Variance (2 components)": "0.72",
            "Reconstruction Error": "0.28"
        }
    elif algorithm in ["CNN", "RNN", "LSTM", "Transformers"]:
        metrics = {
            "Accuracy": "0.91",
            "Loss": "0.245",
            "Training Time": "3.5 hours",
            "Parameters": "4.2 million"
        }
    else:
        metrics = {}
    
    for metric, value in metrics.items():
        st.metric(label=metric, value=value)
    
    # Display some visualizations
    st.subheader("Visualizations")
    
    try:
        if algorithm in ["Linear Regression"]:
            # Generate sample data for visualization
            X_plot = np.linspace(0, 10, 100).reshape(-1, 1)
            y_plot = 2 * X_plot + 1 + np.random.normal(0, 1, (100, 1))
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(X_plot, y_plot, alpha=0.5)
            ax.plot(X_plot, 2 * X_plot + 1, color='red', label='Model Prediction')
            ax.set_xlabel('Feature Value')
            ax.set_ylabel('Target Value')
            ax.set_title('Linear Regression Model Fit')
            ax.legend()
            st.pyplot(fig)
            
            # Residual plot
            residuals = y_plot - (2 * X_plot + 1)
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(X_plot, residuals, alpha=0.5)
            ax.axhline(y=0, color='red', linestyle='-')
            ax.set_xlabel('Feature Value')
            ax.set_ylabel('Residual')
            ax.set_title('Residual Plot')
            st.pyplot(fig)
        
        elif algorithm in ["Logistic Regression", "Decision Tree", "Random Forest", "SVM", "KNN"]:
            # Create a sample confusion matrix
            confusion_matrix = np.array([
                [32, 2, 0],
                [1, 30, 2],
                [0, 3, 30]
            ])
            
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Setosa', 'Versicolor', 'Virginica'],
                       yticklabels=['Setosa', 'Versicolor', 'Virginica'])
            ax.set_xlabel('Predicted Label')
            ax.set_ylabel('True Label')
            ax.set_title('Confusion Matrix')
            st.pyplot(fig)
            
            # ROC curve for binary classification
            if algorithm in ["Logistic Regression", "SVM"]:
                fpr = np.linspace(0, 1, 100)
                tpr = 1 - np.exp(-5 * fpr)  # Simulated ROC curve
                
                fig, ax = plt.subplots(figsize=(10, 8))
                ax.plot(fpr, tpr, label=f'{algorithm} (AUC = 0.92)')
                ax.plot([0, 1], [0, 1], 'k--', label='Random Guess')
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.set_title('ROC Curve')
                ax.legend()
                st.pyplot(fig)
        
        elif algorithm == "K-Means":
            # K-means performance for different k values
            k_values = np.arange(1, 11)
            inertia = 1000 / k_values + 100 + np.random.normal(0, 10, 10)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(k_values, inertia, 'o-')
            ax.set_xlabel('Number of Clusters (k)')
            ax.set_ylabel('Inertia')
            ax.set_title('Elbow Method for Optimal k')
            st.pyplot(fig)
            
            # Cluster visualization
            from sklearn.datasets import make_blobs
            X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
            
            # Assign random cluster labels
            clusters = np.random.randint(0, 4, 300)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            scatter = ax.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis')
            
            # Add centroids
            centroids = np.array([
                [X[clusters == i, 0].mean(), X[clusters == i, 1].mean()]
                for i in range(4)
            ])
            ax.scatter(centroids[:, 0], centroids[:, 1], s=200, marker='*', c='red', label='Centroids')
            
            ax.set_title('K-Means Clustering Result')
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')
            ax.legend()
            st.pyplot(fig)
        
        elif algorithm == "PCA":
            # Scree plot (explained variance)
            n_components = 10
            explained_variance = np.array([0.35, 0.20, 0.15, 0.10, 0.05, 0.05, 0.03, 0.03, 0.02, 0.02])
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(range(1, n_components + 1), explained_variance)
            ax.plot(range(1, n_components + 1), np.cumsum(explained_variance), 'ro-')
            ax.set_xlabel('Principal Components')
            ax.set_ylabel('Explained Variance Ratio')
            ax.set_title('Scree Plot with Cumulative Explained Variance')
            
            # Add cumulative variance labels
            for i, cum_var in enumerate(np.cumsum(explained_variance)):
                ax.annotate(f'{cum_var:.2f}', 
                           (i + 1, cum_var),
                           textcoords="offset points",
                           xytext=(0, 10),
                           ha='center')
            
            st.pyplot(fig)
            
            # Feature contribution
            features = [f'Feature {i+1}' for i in range(10)]
            loadings = np.random.rand(10)
            loadings = loadings / np.sum(loadings)
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.bar(features, loadings)
            ax.set_xlabel('Features')
            ax.set_ylabel('Loading on PC1')
            ax.set_title('Feature Contributions to Principal Component 1')
            plt.xticks(rotation=45)
            st.pyplot(fig)
        
        elif algorithm in ["CNN", "RNN", "LSTM", "Transformers"]:
            # Training history plot
            epochs = np.arange(1, 21)
            train_acc = 1 - 0.9 * np.exp(-0.2 * epochs)
            val_acc = 1 - 0.9 * np.exp(-0.17 * epochs) - 0.05
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(epochs, train_acc, 'b-', label='Training Accuracy')
            ax.plot(epochs, val_acc, 'r-', label='Validation Accuracy')
            ax.set_xlabel('Epochs')
            ax.set_ylabel('Accuracy')
            ax.set_title(f'{algorithm} Training History')
            ax.legend()
            st.pyplot(fig)
            
            # Learning rate vs. performance
            learning_rates = [0.0001, 0.001, 0.01, 0.1, 0.5]
            performances = [0.92, 0.94, 0.91, 0.85, 0.70]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(learning_rates, performances, 'o-')
            ax.set_xscale('log')
            ax.set_xlabel('Learning Rate')
            ax.set_ylabel('Validation Accuracy')
            ax.set_title('Learning Rate vs. Model Performance')
            st.pyplot(fig)
    
    except Exception as e:
        st.error(f"Error rendering visualizations: {str(e)}")
        st.write("This is a demo application showcasing visualizations for machine learning models.")

# Footer
st.markdown("---")
st.markdown("Machine Learning Algorithms Showcase - Created for educational purposes")