from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from db import fetch_table, fetch_query
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

router = APIRouter()

@router.get("/price-prediction/{category}")
def predict_price(
    category: str, 
    features: List[str] = Query(..., description="List of feature columns to use for prediction")
):
    """Predict prices based on product features using linear regression."""
    try:
        # Get the processed data for the category
        table_name = f"processed_{category.lower()}"
        df = fetch_table(table_name)
        
        # Check if all requested features exist
        missing_features = [f for f in features if f not in df.columns]
        if missing_features:
            return {"error": f"Features not found: {missing_features}"}
        
        # Prepare data for model
        X = df[features].apply(pd.to_numeric, errors='coerce')
        y = df['price'].apply(pd.to_numeric, errors='coerce')
        
        # Drop rows with NaN values
        valid_indices = ~(X.isna().any(axis=1) | y.isna())
        X = X[valid_indices]
        y = y[valid_indices]
        
        if len(X) < 10:
            return {"error": "Not enough valid data points for prediction"}
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Evaluate model
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        # Get feature importance
        feature_importance = {
            feature: abs(coef) for feature, coef in zip(features, model.coef_)
        }
        
        return {
            "category": category,
            "features_used": features,
            "sample_size": len(X),
            "r2_train": float(train_score),
            "r2_test": float(test_score),
            "feature_importance": feature_importance,
            "intercept": float(model.intercept_),
            "coefficients": {feature: float(coef) for feature, coef in zip(features, model.coef_)}
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/clustering/{category}")
def cluster_products(
    category: str, 
    features: List[str] = Query(..., description="List of feature columns to use for clustering"),
    n_clusters: int = 3
):
    """Cluster products based on selected features using K-means."""
    try:
        # Get the processed data for the category
        table_name = f"processed_{category.lower()}"
        df = fetch_table(table_name)
        
        # Check if all requested features exist
        missing_features = [f for f in features if f not in df.columns]
        if missing_features:
            return {"error": f"Features not found: {missing_features}"}
        
        # Prepare data for clustering
        X = df[features].apply(pd.to_numeric, errors='coerce')
        
        # Drop rows with NaN values
        X = X.dropna()
        
        if len(X) < n_clusters:
            return {"error": "Not enough valid data points for clustering"}
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Add cluster labels to original data
        result_df = X.copy()
        result_df['cluster'] = clusters
        
        # Get cluster centers (in original scale)
        centers = scaler.inverse_transform(kmeans.cluster_centers_)
        
        # Calculate cluster statistics
        cluster_stats = []
        for i in range(n_clusters):
            cluster_data = result_df[result_df['cluster'] == i]
            stats = {
                "cluster_id": i,
                "size": len(cluster_data),
                "percentage": round(len(cluster_data) / len(result_df) * 100, 2),
                "center": {feature: float(centers[i, j]) for j, feature in enumerate(features)},
                "min": {feature: float(cluster_data[feature].min()) for feature in features},
                "max": {feature: float(cluster_data[feature].max()) for feature in features},
                "mean": {feature: float(cluster_data[feature].mean()) for feature in features}
            }
            cluster_stats.append(stats)
        
        return {
            "category": category,
            "features_used": features,
            "n_clusters": n_clusters,
            "sample_size": len(X),
            "clusters": cluster_stats,
            "inertia": float(kmeans.inertia_)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/association-rules/{category}")
def get_association_rules(category: str, min_support: float = 0.1):
    """Extract association rules from product features."""
    try:
        from mlxtend.frequent_patterns import apriori, association_rules
        
        # Get the processed data for the category
        table_name = f"processed_{category.lower()}"
        df = fetch_table(table_name)
        
        # Identify categorical columns (excluding IDs, prices, etc.)
        exclude_cols = ['id', 'asin', 'price', 'rating', 'review_count', 'title', 'url', 'brand', 'category']
        categorical_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype == 'object']
        
        if not categorical_cols:
            return {"error": "No categorical columns found for association rules"}
        
        # Prepare data for association rules
        # Convert categorical data to one-hot encoding
        encoded_df = pd.get_dummies(df[categorical_cols], prefix_sep='=')
        
        # Apply Apriori algorithm to find frequent itemsets
        frequent_itemsets = apriori(encoded_df, min_support=min_support, use_colnames=True)
        
        if frequent_itemsets.empty:
            return {
                "category": category,
                "warning": f"No frequent itemsets found with min_support={min_support}. Try lowering the support threshold."
            }
        
        # Generate association rules
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
        
        if rules.empty:
            return {
                "category": category,
                "warning": "No association rules found with confidence threshold of 0.5"
            }
        
        # Format the results
        formatted_rules = []
        for _, rule in rules.iterrows():
            antecedents = list(rule['antecedents'])
            consequents = list(rule['consequents'])
            formatted_rules.append({
                "antecedents": antecedents,
                "consequents": consequents,
                "support": float(rule['support']),
                "confidence": float(rule['confidence']),
                "lift": float(rule['lift'])
            })
        
        return {
            "category": category,
            "min_support": min_support,
            "features_used": categorical_cols,
            "num_rules": len(formatted_rules),
            "rules": formatted_rules[:20]  # Limit to top 20 rules
        }
    except ImportError:
        return {"error": "Required packages not installed. Install mlxtend with 'pip install mlxtend'"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/outlier-detection/{category}")
def detect_outliers(
    category: str,
    feature: str = Query(..., description="Feature to analyze for outliers"),
    method: str = "zscore",
    threshold: float = 3.0
):
    """Detect outliers in a specific feature using various methods."""
    try:
        # Get the processed data for the category
        table_name = f"processed_{category.lower()}"
        df = fetch_table(table_name)
        
        # Check if feature exists
        if feature not in df.columns:
            return {"error": f"Feature '{feature}' not found in category '{category}'"}
        
        # Convert to numeric and drop NaN values
        values = pd.to_numeric(df[feature], errors='coerce').dropna()
        
        if len(values) < 10:
            return {"error": f"Not enough valid data points for feature '{feature}'"}
        
        outliers = []
        
        if method == "zscore":
            # Z-score method
            z_scores = np.abs((values - values.mean()) / values.std())
            outlier_indices = np.where(z_scores > threshold)[0]
            outliers = values.iloc[outlier_indices].tolist()
            
        elif method == "iqr":
            # IQR method
            Q1 = values.quantile(0.25)
            Q3 = values.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outlier_indices = np.where((values < lower_bound) | (values > upper_bound))[0]
            outliers = values.iloc[outlier_indices].tolist()
        
        else:
            return {"error": f"Unsupported method: {method}. Use 'zscore' or 'iqr'"}
        
        # Get the corresponding products for outliers
        outlier_products = []
        if outlier_indices.size > 0:
            outlier_df = df.iloc[outlier_indices]
            outlier_products = outlier_df[['asin', 'title', feature, 'price']].to_dict(orient='records')
        
        return {
            "category": category,
            "feature": feature,
            "method": method,
            "threshold": threshold,
            "total_products": len(values),
            "outlier_count": len(outliers),
            "outlier_percentage": round(len(outliers) / len(values) * 100, 2),
            "min_value": float(values.min()),
            "max_value": float(values.max()),
            "mean_value": float(values.mean()),
            "median_value": float(values.median()),
            "outlier_products": outlier_products[:20]  # Limit to top 20 outliers
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/price-forecast/{category}")
def forecast_price_trends(category: str, forecast_periods: int = 3):
    """Forecast price trends for a category using time series analysis."""
    try:
        from statsmodels.tsa.arima.model import ARIMA
        
        # Get the time series data for the category
        query = f"""
        SELECT DATE(created_at) as date, AVG(price) as avg_price
        FROM products
        WHERE category = '{category}' AND price IS NOT NULL
        GROUP BY DATE(created_at)
        ORDER BY date
        """
        df = fetch_query(query)
        
        if len(df) < 7:
            return {"error": f"Not enough time series data for category '{category}'. Need at least 7 data points."}
        
        # Set date as index
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        # Fit ARIMA model
        model = ARIMA(df['avg_price'], order=(1, 1, 1))
        model_fit = model.fit()
        
        # Forecast future prices
        forecast = model_fit.forecast(steps=forecast_periods)
        forecast_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=forecast_periods)
        
        # Prepare historical data for response
        historical = [
            {"date": date.strftime('%Y-%m-%d'), "avg_price": float(price)}
            for date, price in zip(df.index, df['avg_price'])
        ]
        
        # Prepare forecast data for response
        forecast_data = [
            {"date": date.strftime('%Y-%m-%d'), "avg_price": float(price)}
            for date, price in zip(forecast_dates, forecast)
        ]
        
        return {
            "category": category,
            "historical_data_points": len(historical),
            "forecast_periods": forecast_periods,
            "historical": historical,
            "forecast": forecast_data,
            "current_avg_price": float(df['avg_price'].iloc[-1]),
            "forecast_trend": "increasing" if forecast[-1] > df['avg_price'].iloc[-1] else "decreasing"
        }
    except ImportError:
        return {"error": "Required packages not installed. Install statsmodels with 'pip install statsmodels'"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/feature-importance/{category}")
def get_feature_importance(category: str):
    """Determine which features have the most impact on product price."""
    try:
        from sklearn.ensemble import RandomForestRegressor
        
        # Get the processed data for the category
        table_name = f"processed_{category.lower()}"
        df = fetch_table(table_name)
        
        # Identify potential feature columns (excluding non-numeric and identifier columns)
        exclude_cols = ['id', 'asin', 'title', 'url', 'brand', 'category', 'price']
        feature_cols = []
        
        for col in df.columns:
            if col not in exclude_cols:
                # Try to convert to numeric
                numeric_series = pd.to_numeric(df[col], errors='coerce')
                # If at least 50% of values are valid numbers, include the feature
                if numeric_series.notna().sum() >= len(df) * 0.5:
                    feature_cols.append(col)
        
        if not feature_cols:
            return {"error": f"No suitable numeric features found for category '{category}'"}
        
        # Prepare data for model
        X = df[feature_cols].apply(pd.to_numeric, errors='coerce')
        y = df['price'].apply(pd.to_numeric, errors='coerce')
        
        # Drop rows with NaN values
        valid_indices = ~(X.isna().any(axis=1) | y.isna())
        X = X[valid_indices]
        y = y[valid_indices]
        
        if len(X) < 10:
            return {"error": "Not enough valid data points for analysis"}
        
        # Train Random Forest model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # Get feature importance
        importance = model.feature_importances_
        
        # Create sorted feature importance dictionary
        feature_importance = {
            feature: float(importance[i])
            for i, feature in enumerate(feature_cols)
        }
        
        # Sort by importance
        sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        return {
            "category": category,
            "features_analyzed": len(feature_cols),
            "sample_size": len(X),
            "feature_importance": dict(sorted_importance),
            "top_features": dict(sorted_importance[:5])
        }
    except ImportError:
        return {"error": "Required packages not installed. Install scikit-learn with 'pip install scikit-learn'"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

