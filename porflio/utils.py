import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_model(model_path):
    """
    Load a machine learning model from the given path.
    
    Parameters:
    -----------
    model_path : str
        Path to the saved model file
        
    Returns:
    --------
    model : object
        The loaded model object
    """
    try:
        # Check if file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        # Load the model based on file extension
        file_extension = os.path.splitext(model_path)[1].lower()
        
        if file_extension == '.pkl':
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
        elif file_extension == '.h5':
            # For deep learning models
            try:
                import tensorflow as tf
                model = tf.keras.models.load_model(model_path)
            except ImportError:
                raise ImportError("TensorFlow is required to load .h5 models")
        else:
            raise ValueError(f"Unsupported model file format: {file_extension}")
        
        return model
    
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None


def preprocess_data(data, model_type, scaler=None):
    """
    Preprocess input data according to model requirements.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input data to preprocess
    model_type : str
        Type of model to preprocess for ('classification', 'regression', 'clustering', etc.)
    scaler : sklearn.preprocessing object, optional
        Scaler to use for feature scaling. If None, a new scaler will be created.
        
    Returns:
    --------
    processed_data : pandas.DataFrame or numpy.ndarray
        Preprocessed data ready for model input
    scaler : sklearn.preprocessing object
        The scaler used for preprocessing (for inverse transforms later)
    """
    try:
        # Handle missing values
        data = data.copy()
        data.fillna(data.mean(numeric_only=True), inplace=True)
        
        # Feature scaling for most models
        if model_type in ['classification', 'regression', 'clustering', 'svm', 'knn']:
            if scaler is None:
                scaler = StandardScaler()
                processed_data = scaler.fit_transform(data)
            else:
                processed_data = scaler.transform(data)
        else:
            # For tree-based models, scaling may not be necessary
            processed_data = data.values if isinstance(data, pd.DataFrame) else data
            
        return processed_data, scaler
    
    except Exception as e:
        print(f"Error preprocessing data: {str(e)}")
        return data, scaler
