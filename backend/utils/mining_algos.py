import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def apply_kmeans(
    data: pd.DataFrame, 
    features: List[str], 
    n_clusters: int = 3
) -> Tuple[np.ndarray, KMeans, StandardScaler]:
    """
    Apply K-means clustering to the data.
    
    Args:
        data: DataFrame containing the data
        features: List of feature columns to use
        n_clusters: Number of clusters to create
        
    Returns:
        Tuple containing (cluster_labels, kmeans_model, scaler)
    """
    # Extract features
    X = data[features].apply(pd.to_numeric, errors='coerce')
    
    # Drop rows with NaN values
    X = X.dropna()
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    return clusters, kmeans, scaler

def extract_association_rules(
    data: pd.DataFrame,
    min_support: float = 0.1,
    min_confidence: float = 0.5
) -> Dict:
    """
    Extract association rules from categorical data.
    
    Args:
        data: DataFrame containing the data
        min_support: Minimum support threshold
        min_confidence: Minimum confidence threshold
        
    Returns:
        Dictionary with frequent itemsets and rules
    """
    try:
        from mlxtend.frequent_patterns import apriori, association_rules
        
        # Identify categorical columns
        exclude_cols = ['id', 'asin', 'price', 'rating', 'review_count', 'title', 'url']
        categorical_cols = [col for col in data.columns if col not in exclude_cols and data[col].dtype == 'object']
        
        if not categorical_cols:
            return {"error": "No categorical columns found"}
        
        # Convert categorical data to one-hot encoding
        encoded_df = pd.get_dummies(data[categorical_cols], prefix_sep='=')
        
        # Apply Apriori algorithm
        frequent_itemsets = apriori(encoded_df, min_support=min_support, use_colnames=True)
        
        if frequent_itemsets.empty:
            return {"warning": f"No frequent itemsets found with min_support={min_support}"}
        
        # Generate rules
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
        
        return {
            "frequent_itemsets": frequent_itemsets,
            "rules": rules,
            "features_used": categorical_cols
        }
    except ImportError:
        return {"error": "Required packages not installed. Install mlxtend with 'pip install mlxtend'"}

def detect_outliers(
    data: pd.Series,
    method: str = "zscore",
    threshold: float = 3.0
) -> np.ndarray:
    """
    Detect outliers in a data series.
    
    Args:
        data: Series of numeric values
        method: Method to use ('zscore' or 'iqr')
        threshold: Threshold for outlier detection
        
    Returns:
        Array of indices of outliers
    """
    if method == "zscore":
        # Z-score method
        z_scores = np.abs((data - data.mean()) / data.std())
        return np.where(z_scores > threshold)[0]
        
    elif method == "iqr":
        # IQR method
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        return np.where((data < lower_bound) | (data > upper_bound))[0]
    
    else:
        raise ValueError(f"Unsupported method: {method}. Use 'zscore' or 'iqr'")

def price_prediction_model(
    df: pd.DataFrame,
    features: List[str],
    test_size: float = 0.2
) -> Dict:
    """
    Build a price prediction model.
    
    Args:
        df: DataFrame containing the data
        features: List of feature columns to use
        test_size: Proportion of data to use for testing
        
    Returns:
        Dictionary with model results
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    
    # Prepare data
    X = df[features].apply(pd.to_numeric, errors='coerce')
    y = df['price'].apply(pd.to_numeric, errors='coerce')
    
    # Drop rows with NaN values
    valid_indices = ~(X.isna().any(axis=1) | y.isna())
    X = X[valid_indices]
    y = y[valid_indices]
    
    if len(X) < 10:
        return {"error": "Not enough valid data points for prediction"}
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Evaluate model
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    return {
        "model": model,
        "train_score": train_score,
        "test_score": test_score,
        "feature_importance": {
            feature: abs(coef) for feature, coef in zip(features, model.coef_)
        },
        "coefficients": {
            feature: coef for feature, coef in zip(features, model.coef_)
        },
        "intercept": model.intercept_
    }
