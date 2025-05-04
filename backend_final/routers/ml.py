# routers/ml.py

from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from database import get_db
from models import products as prod_models, specifications as spec_models
from typing import List, Dict, Optional, Union, Any
import pandas as pd
import numpy as np
import logging
from io import BytesIO
import base64
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.svm import SVR, SVC
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, silhouette_score
from sklearn.metrics import confusion_matrix, classification_report
from mlxtend.frequent_patterns import apriori, association_rules

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

# ===== Pydantic Models =====

class ProductFeature(BaseModel):
    product_id: int
    price: Optional[float] = None
    rating: Optional[float] = None
    review_count: Optional[int] = None
    category: Optional[str] = None
    brand: Optional[str] = None
    specs: Optional[Dict[str, str]] = None
    
class MLRequest(BaseModel):
    features: List[ProductFeature]
    target_column: str
    algorithm: str
    params: Optional[Dict[str, Any]] = {}
    test_size: Optional[float] = 0.2

class AssociationRuleRequest(BaseModel):
    min_support: float = Field(0.1, ge=0.0, le=1.0)
    min_confidence: float = Field(0.5, ge=0.0, le=1.0)
    min_lift: float = Field(1.0, ge=0.0)
    metric: str = "lift"
    category: Optional[str] = None
    
class ClusteringRequest(BaseModel):
    algorithm: str
    n_clusters: Optional[int] = 3
    params: Optional[Dict[str, Any]] = {}
    features: List[str]

# ===== Helper Functions =====

def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string for HTML display"""
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_str

def preprocess_data(df: pd.DataFrame, categorical_cols=None, numerical_cols=None) -> pd.DataFrame:
    """
    Preprocess dataframe for ML algorithms
    """
    try:
        # Handle categorical columns
        if categorical_cols is None:
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

        # Handle numerical columns
        if numerical_cols is None:
            numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            # Remove product_id from numerical columns if it exists
            if 'product_id' in numerical_cols:
                numerical_cols.remove('product_id')

        # Remove the specs column as it's a dictionary and handle separately
        if 'specs' in df.columns:
            df = df.drop(columns=['specs'], errors='ignore')

        # Handle missing values in numerical columns
        for col in numerical_cols:
            if col in df.columns:
                df.loc[:, col] = df[col].fillna(df[col].median())

        # Encode categorical columns
        for col in categorical_cols:
            if col in df.columns:
                # Skip columns with too many unique values or all NaN
                if df[col].nunique() > 100 or df[col].isna().all():
                    df = df.drop(columns=[col], errors='ignore')
                    continue

                # Fill missing values with the most common value
                df.loc[:, col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "unknown")

                # Use label encoding for categorical columns
                le = LabelEncoder()
                df.loc[:, col] = le.fit_transform(df[col])

        # Replace any remaining NaN values with 0
        df = df.fillna(0)

        # Log if there are still NaN values
        if df.isnull().values.any():
            logger.error(f"DataFrame still contains NaN values after preprocessing: {df.isnull().sum()}")

        return df

    except Exception as e:
        logger.error(f"Error preprocessing data: {str(e)}")
        raise ValueError(f"Error preprocessing data: {str(e)}")


def check_algorithm_applicability(df: pd.DataFrame, algorithm: str, target_col: str = None) -> Dict[str, Any]:
    """
    Check if an algorithm is applicable to the given dataset and provide reasons if not
    """
    result = {"applicable": True, "reasons": []}
    
    # Minimum data requirement
    if len(df) < 10:
        result["applicable"] = False
        result["reasons"].append("Dataset has fewer than 10 samples, which is too small for reliable ML modeling")
    
    # Regression specific checks
    regression_algorithms = ["linear_regression", "ridge", "lasso", "svr", "random_forest_regressor", "gradient_boosting"]
    if algorithm in regression_algorithms:
        if target_col not in df.columns:
            result["applicable"] = False
            result["reasons"].append(f"Target column '{target_col}' not found in dataset")
        elif not np.issubdtype(df[target_col].dtype, np.number):
            result["applicable"] = False
            result["reasons"].append(f"Target column '{target_col}' must be numeric for regression")
        elif df[target_col].nunique() < 5:
            result["applicable"] = False
            result["reasons"].append(f"Target column '{target_col}' has fewer than 5 unique values, which may indicate it's not suitable for regression")
    
    # Classification specific checks
    classification_algorithms = ["logistic_regression", "random_forest_classifier", "svc"]
    if algorithm in classification_algorithms:
        if target_col not in df.columns:
            result["applicable"] = False
            result["reasons"].append(f"Target column '{target_col}' not found in dataset")
        elif df[target_col].nunique() < 2:
            result["applicable"] = False
            result["reasons"].append(f"Target column '{target_col}' has fewer than 2 classes, which is not suitable for classification")
        elif df[target_col].nunique() > len(df) * 0.5:
            result["applicable"] = False
            result["reasons"].append(f"Target column '{target_col}' has too many unique values relative to the dataset size, which may indicate it's not suitable for classification")
    
    # Clustering specific checks
    clustering_algorithms = ["kmeans", "dbscan", "hierarchical"]
    if algorithm in clustering_algorithms:
        if len(df.columns) < 2:
            result["applicable"] = False
            result["reasons"].append("At least 2 feature columns are required for clustering")
    
    return result

def get_algorithm(algorithm_name: str, params: Dict[str, Any] = None):
    """Get the appropriate scikit-learn algorithm instance based on name and parameters"""
    
    if params is None:
        params = {}
    
    # Regression algorithms
    if algorithm_name == "linear_regression":
        return LinearRegression(**params)
    elif algorithm_name == "ridge":
        return Ridge(**params)
    elif algorithm_name == "lasso":
        return Lasso(**params)
    elif algorithm_name == "svr":
        return SVR(**params)
    elif algorithm_name == "random_forest_regressor":
        return RandomForestRegressor(**params)
    elif algorithm_name == "gradient_boosting":
        return GradientBoostingRegressor(**params)
    
    # Classification algorithms
    elif algorithm_name == "logistic_regression":
        return LogisticRegression(**params)
    elif algorithm_name == "random_forest_classifier":
        return RandomForestClassifier(**params)
    elif algorithm_name == "svc":
        return SVC(**params)
    
    # Clustering algorithms
    elif algorithm_name == "kmeans":
        return KMeans(**params)
    elif algorithm_name == "dbscan":
        return DBSCAN(**params)
    elif algorithm_name == "hierarchical":
        return AgglomerativeClustering(**params)
    
    else:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")

def visualize_model_results(model_type, model, X, y=None, y_pred=None, feature_names=None):
    """Generate appropriate visualizations for the model results"""
    
    figures = {}
    
    # Set the style
    sns.set(style="whitegrid")
    
    try:
        # Regression visualizations
        if model_type in ["linear_regression", "ridge", "lasso", "svr", "random_forest_regressor", "gradient_boosting"]:
            # 1. Actual vs Predicted Values
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(y, y_pred, alpha=0.5)
            ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
            ax.set_xlabel('Actual')
            ax.set_ylabel('Predicted')
            ax.set_title('Actual vs Predicted Values')
            figures['actual_vs_predicted'] = fig_to_base64(fig)
            
            # 2. Residuals Plot
            residuals = y - y_pred
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(y_pred, residuals, alpha=0.5)
            ax.axhline(y=0, color='r', linestyle='-')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Residuals')
            ax.set_title('Residuals Plot')
            figures['residuals'] = fig_to_base64(fig)
            
            # 3. Feature Importance (for tree-based models)
            if hasattr(model, 'feature_importances_') and feature_names is not None:
                importances = pd.Series(model.feature_importances_, index=feature_names)
                fig, ax = plt.subplots(figsize=(10, max(6, len(feature_names) * 0.3)))
                importances.sort_values().plot(kind='barh', ax=ax)
                ax.set_title('Feature Importance')
                figures['feature_importance'] = fig_to_base64(fig)
                
        # Classification visualizations
        elif model_type in ["logistic_regression", "random_forest_classifier", "svc"]:
            # 1. Confusion Matrix
            cm = confusion_matrix(y, y_pred)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title('Confusion Matrix')
            figures['confusion_matrix'] = fig_to_base64(fig)
            
            # 2. Feature Importance for Random Forest
            if hasattr(model, 'feature_importances_') and feature_names is not None:
                importances = pd.Series(model.feature_importances_, index=feature_names)
                fig, ax = plt.subplots(figsize=(10, max(6, len(feature_names) * 0.3)))
                importances.sort_values().plot(kind='barh', ax=ax)
                ax.set_title('Feature Importance')
                figures['feature_importance'] = fig_to_base64(fig)
                
        # Clustering visualizations
        elif model_type in ["kmeans", "dbscan", "hierarchical"]:
            # We need at least 2D data to visualize clusters
            if X.shape[1] >= 2:
                # Use the first two features
                fig, ax = plt.subplots(figsize=(10, 8))
                scatter = ax.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', alpha=0.5)
                ax.set_title('Clusters')
                plt.colorbar(scatter, ax=ax, label='Cluster')
                figures['clusters_2d'] = fig_to_base64(fig)
            
            # If we have more than 2 features, use PCA to visualize
            if X.shape[1] > 2:
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X)
                
                fig, ax = plt.subplots(figsize=(10, 8))
                scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y_pred, cmap='viridis', alpha=0.5)
                ax.set_title('Clusters (PCA)')
                plt.colorbar(scatter, ax=ax, label='Cluster')
                figures['clusters_pca'] = fig_to_base64(fig)
    except Exception as e:
        logger.error(f"Error generating visualizations: {str(e)}")
    
    return figures

def visualize_association_rules(rules_df):
    """Generate visualizations for association rules"""
    figures = {}
    
    try:
        # 1. Scatter plot of support vs confidence
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(rules_df['support'], rules_df['confidence'], alpha=0.5, c=rules_df['lift'], cmap='viridis')
        ax.set_xlabel('Support')
        ax.set_ylabel('Confidence')
        ax.set_title('Support vs Confidence')
        plt.colorbar(scatter, ax=ax, label='Lift')
        figures['support_vs_confidence'] = fig_to_base64(fig)
        
        # 2. Distribution of lift values
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(rules_df['lift'], bins=20, kde=True, ax=ax)
        ax.set_xlabel('Lift')
        ax.set_title('Distribution of Lift Values')
        figures['lift_distribution'] = fig_to_base64(fig)
        
        # 3. Top rules by lift
        top_rules = rules_df.sort_values('lift', ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Convert antecedents and consequents to strings for display
        top_rules['antecedents_str'] = top_rules['antecedents'].apply(lambda x: ', '.join(list(x)))
        top_rules['consequents_str'] = top_rules['consequents'].apply(lambda x: ', '.join(list(x)))
        
        # Create the plot
        y_pos = range(len(top_rules))
        ax.barh(y_pos, top_rules['lift'])
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"{a} → {c}" for a, c in zip(top_rules['antecedents_str'], top_rules['consequents_str'])])
        ax.set_xlabel('Lift')
        ax.set_title('Top 10 Rules by Lift')
        
        # Adjust layout for better display
        plt.tight_layout()
        figures['top_rules'] = fig_to_base64(fig)
    except Exception as e:
        logger.error(f"Error generating association rule visualizations: {str(e)}")
    
    return figures

# ===== API Endpoints =====

@router.post("/supervised-learning")
async def run_supervised_learning(request: MLRequest):
    """
    Endpoint for supervised learning algorithms (regression and classification)
    """
    try:
        # Convert request to DataFrame
        df = pd.DataFrame([item.dict() for item in request.features])
        logger.info(f"Received data with shape: {df.shape}")
        
        # Check if dataset is empty
        if df.empty:
            raise HTTPException(status_code=400, detail="Empty dataset provided")
        
        # Expand specs dictionary if it exists
        if 'specs' in df.columns and not df['specs'].isna().all():
            logger.info("Expanding specs dictionary")
            specs_df = pd.json_normalize(df['specs'].dropna())
            
            # Only include columns that appear in at least 30% of rows
            threshold = 0.3 * len(specs_df)
            specs_cols_to_keep = [col for col in specs_df.columns if specs_df[col].count() >= threshold]
            specs_df = specs_df[specs_cols_to_keep]
            
            # Remove the specs column and join with expanded specs
            if not specs_df.empty:
                df = df.drop(columns=['specs'], errors='ignore')
                specs_df.index = df.index[:len(specs_df)]
                df = pd.concat([df, specs_df], axis=1)
        
        # Validate target column exists
        if request.target_column not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Target column '{request.target_column}' not found in dataset. Available columns: {', '.join(df.columns)}"
            )
        
        # Check if algorithm is applicable
        applicability = check_algorithm_applicability(df, request.algorithm, request.target_column)
        if not applicability["applicable"]:
            return {
                "success": False,
                "message": "The selected algorithm is not applicable to this dataset",
                "reasons": applicability["reasons"]
            }
        
        # Separate features and target
        X = df.drop(columns=[request.target_column, 'product_id'], errors='ignore')
        y = df[request.target_column]
        
        # Preprocess data
        X = preprocess_data(X)
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=request.test_size, random_state=42)
        
        # Get feature names for visualization
        feature_names = X.columns.tolist()
        
        # Get the algorithm
        model = get_algorithm(request.algorithm, request.params)
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Evaluate the model
        results = {}
        
        # Classification metrics
        if request.algorithm in ["logistic_regression", "random_forest_classifier", "svc"]:
            accuracy = accuracy_score(y_test, y_pred)
            class_report = classification_report(y_test, y_pred, output_dict=True)
            
            results = {
                "success": True,
                "metrics": {
                    "accuracy": accuracy,
                    "precision": class_report['weighted avg']['precision'],
                    "recall": class_report['weighted avg']['recall'],
                    "f1_score": class_report['weighted avg']['f1-score']
                },
                "detailed_report": class_report
            }
        
        # Regression metrics
        else:
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
            
            results = {
                "success": True,
                "metrics": {
                    "mse": mse,
                    "r2": r2,
                    "cv_r2_mean": cv_scores.mean(),
                    "cv_r2_std": cv_scores.std()
                }
            }
        
        # Generate visualizations
        visualizations = visualize_model_results(
            request.algorithm, 
            model, 
            X_test.values, 
            y_test, 
            y_pred,
            feature_names
        )
        
        results["visualizations"] = visualizations
        
        # Feature importance if available
        if hasattr(model, 'feature_importances_'):
            results["feature_importance"] = dict(zip(feature_names, model.feature_importances_.tolist()))
        
        return results
    
    except ValueError as ve:
        logger.error(f"Value error in supervised learning: {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Error in supervised learning: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/clustering")
async def run_clustering(request: ClusteringRequest, db: Session = Depends(get_db)):
    """
    Endpoint for clustering algorithms
    """
    try:
        # Get product data from database
        products = db.query(prod_models.Product).all()

        if not products:
            raise HTTPException(status_code=404, detail="No product data available")

        # Convert to DataFrame
        df = pd.DataFrame([{
            "product_id": p.product_id,
            "price": p.price,
            "rating": p.rating,
            "review_count": p.review_count,
            "category": p.category,
            "brand": p.brand
        } for p in products])

        # Filter features
        available_features = [col for col in request.features if col in df.columns]

        if not available_features:
            raise HTTPException(
                status_code=400,
                detail=f"None of the requested features are available. Available features: {', '.join(df.columns)}"
            )

        # Subset the data
        X = df[available_features]

        # Preprocess data
        X = preprocess_data(X)

        # Check for NaN values after preprocessing
        if X.isnull().values.any():
            logger.error(f"Data contains NaN values after preprocessing: {X.isnull().sum()}")
            raise HTTPException(status_code=400, detail="Data contains NaN values after preprocessing")

        # Set parameters
        params = request.params.copy()

        # Add n_clusters parameter if applicable
        if request.algorithm in ["kmeans", "hierarchical"]:
            params["n_clusters"] = request.n_clusters

        # Get clustering algorithm
        model = get_algorithm(request.algorithm, params)

        # Fit the model
        labels = model.fit_predict(X)

        # Handle NaN values in labels (e.g., DBSCAN noise points)
        if np.isnan(labels).any():
            labels = np.where(np.isnan(labels), -1, labels)

        # Check for NaN values in labels after handling
        if np.isnan(labels).any():
            logger.error(f"Cluster labels contain NaN values after handling: {np.isnan(labels).sum()}")
            raise HTTPException(status_code=500, detail="Cluster labels contain NaN values after handling")

        # Evaluate clusters if possible
        metrics = {}
        if request.algorithm in ["kmeans", "hierarchical"] and len(set(labels)) >= 2:
            silhouette = silhouette_score(X, labels)
            metrics["silhouette_score"] = silhouette

        # Generate visualizations
        visualizations = visualize_model_results(request.algorithm, model, X.values, None, labels, X.columns.tolist())

        # Prepare cluster statistics
        cluster_stats = {}
        if len(set(labels)) > 1:  # Only if we have valid clusters
            df['cluster'] = labels

            # Compute statistics for each cluster
            for cluster_id in set(labels):
                if cluster_id == -1:  # DBSCAN noise points
                    continue

                cluster_data = df[df['cluster'] == cluster_id]

                # Calculate statistics for numeric columns
                numeric_cols = cluster_data.select_dtypes(include=['int64', 'float64']).columns
                stats = {}

                for col in numeric_cols:
                    if col != 'cluster' and col != 'product_id':
                        stats[col] = {
                            "mean": float(cluster_data[col].mean()),
                            "median": float(cluster_data[col].median()),
                            "min": float(cluster_data[col].min()),
                            "max": float(cluster_data[col].max()),
                            "count": int(cluster_data[col].count())
                        }

                cluster_stats[str(cluster_id)] = stats

        # Return results with cluster assignments
        result = {
            "success": True,
            "algorithm": request.algorithm,
            "cluster_assignments": [int(label) for label in labels],
            "num_clusters": len(set([l for l in labels if l != -1])),  # Exclude noise points
            "product_ids": df["product_id"].tolist(),
            "metrics": metrics,
            "cluster_statistics": cluster_stats,
            "visualizations": visualizations
        }

        return result

    except ValueError as ve:
        logger.error(f"Value error in clustering: {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Error in clustering: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/association-rules")
async def run_association_rules(request: AssociationRuleRequest, db: Session = Depends(get_db)):
    """
    Endpoint for association rule mining
    """
    try:
        # Query specifications from the database
        query = db.query(
            spec_models.Specification.product_id,
            spec_models.Specification.spec_name,
            spec_models.Specification.spec_value
        )
        
        # Filter by category if provided
        if request.category:
            product_ids = db.query(prod_models.Product.product_id).filter(
                prod_models.Product.category == request.category
            ).subquery()
            query = query.filter(spec_models.Specification.product_id.in_(product_ids))
        
        specs = query.all()
        
        if not specs:
            raise HTTPException(status_code=404, detail="No specification data available for mining")
        
        # Create a DataFrame from specifications
        specs_df = pd.DataFrame([(p_id, f"{name}={value}") for p_id, name, value in specs], 
                                columns=["product_id", "feature"])
        
        # Create a binary (one-hot encoded) representation for apriori
        one_hot = pd.crosstab(specs_df["product_id"], specs_df["feature"])
        
        # Check if we have enough data
        if one_hot.shape[0] < 10 or one_hot.shape[1] < 2:
            return {
                "success": False,
                "message": "Insufficient data for association rule mining",
                "details": f"Need at least 10 products and 2 features, got {one_hot.shape[0]} products and {one_hot.shape[1]} features"
            }
        
        # Run apriori algorithm
        frequent_itemsets = apriori(one_hot, 
                                   min_support=request.min_support, 
                                   use_colnames=True)
        
        # If no frequent itemsets were found, return early
        if frequent_itemsets.empty:
            return {
                "success": False,
                "message": "No frequent itemsets found with the current support threshold",
                "suggestion": "Try lowering the min_support value"
            }
        
        # Generate association rules
        rules = association_rules(
            frequent_itemsets, 
            metric=request.metric, 
            min_threshold=request.min_confidence
        )
        
        # If no rules were found, return early
        if rules.empty:
            return {
                "success": False,
                "message": "No association rules found with the current thresholds",
                "suggestion": "Try lowering the min_confidence or min_lift values"
            }
        
        # Filter by minimum lift if specified
        if request.min_lift > 1.0:
            rules = rules[rules['lift'] >= request.min_lift]
            
            if rules.empty:
                return {
                    "success": False,
                    "message": "No association rules found with the current lift threshold",
                    "suggestion": "Try lowering the min_lift value"
                }
        
        # Convert rules to a more readable format
        readable_rules = []
        for _, rule in rules.iterrows():
            antecedents = list(rule['antecedents'])
            consequents = list(rule['consequents'])
            
            readable_rules.append({
                "antecedents": antecedents,
                "consequents": consequents,
                "support": rule['support'],
                "confidence": rule['confidence'],
                "lift": rule['lift']
            })
        
        # Generate visualizations
        visualizations = visualize_association_rules(rules)
        
        return {
            "success": True,
            "num_rules": len(readable_rules),
            "rules": readable_rules,
            "visualizations": visualizations
        }
        
    except ValueError as ve:
        logger.error(f"Value error in association rules: {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Error in association rules: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/available-algorithms")
async def get_available_algorithms():
    """
    Return the list of available algorithms with their parameters and descriptions
    """
    return {
        "success": True,
        "algorithms": {
            "supervised": {
                "regression": [
                    {
                        "name": "linear_regression",
                        "display_name": "Linear Regression",
                        "description": "Simple linear regression model that finds the line of best fit",
                        "parameters": []
                    },
                    {
                        "name": "ridge",
                        "display_name": "Ridge Regression",
                        "description": "Linear regression with L2 regularization",
                        "parameters": [
                            {"name": "alpha", "type": "float", "default": 1.0, "description": "Regularization strength"}
                        ]
                    },
                    {
                        "name": "lasso",
                        "display_name": "Lasso Regression",
                        "description": "Linear regression with L1 regularization (feature selection)",
                        "parameters": [
                            {"name": "alpha", "type": "float", "default": 1.0, "description": "Regularization strength"}
                        ]
                    },
                    {
                        "name": "svr",
                        "display_name": "Support Vector Regression",
                        "description": "Support Vector Machine for regression tasks",
                        "parameters": [
                            {"name": "C", "type": "float", "default": 1.0, "description": "Regularization parameter"},
                            {"name": "kernel", "type": "string", "default": "rbf", "description": "Kernel type (rbf, linear, poly, sigmoid)"},
                            {"name": "gamma", "type": "string", "default": "scale", "description": "Kernel coefficient"}
                        ]
                    },
                    {
                        "name": "random_forest_regressor",
                        "display_name": "Random Forest Regressor",
                        "description": "Ensemble of decision trees for regression",
                        "parameters": [
                            {"name": "n_estimators", "type": "int", "default": 100, "description": "Number of trees"},
                            {"name": "max_depth", "type": "int", "default": None, "description": "Maximum depth of trees"},
                            {"name": "min_samples_split", "type": "int", "default": 2, "description": "Minimum samples required to split"}
                        ]
                    },
                    {
                        "name": "gradient_boosting",
                        "display_name": "Gradient Boosting Regressor",
                        "description": "Boosting ensemble of decision trees for regression",
                        "parameters": [
                            {"name": "n_estimators", "type": "int", "default": 100, "description": "Number of boosting stages"},
                            {"name": "learning_rate", "type": "float", "default": 0.1, "description": "Learning rate"},
                            {"name": "max_depth", "type": "int", "default": 3, "description": "Maximum depth of trees"}
                        ]
                    }
                ],
                "classification": [
                    {
                        "name": "logistic_regression",
                        "display_name": "Logistic Regression",
                        "description": "Linear model for classification",
                        "parameters": [
                            {"name": "C", "type": "float", "default": 1.0, "description": "Inverse regularization strength"},
                            {"name": "penalty", "type": "string", "default": "l2", "description": "Penalty norm (l1, l2, elasticnet, none)"},
                            {"name": "solver", "type": "string", "default": "lbfgs", "description": "Algorithm for optimization"}
                        ]
                    },
                    {
                        "name": "random_forest_classifier",
                        "display_name": "Random Forest Classifier",
                        "description": "Ensemble of decision trees for classification",
                        "parameters": [
                            {"name": "n_estimators", "type": "int", "default": 100, "description": "Number of trees"},
                            {"name": "max_depth", "type": "int", "default": None, "description": "Maximum depth of trees"},
                            {"name": "min_samples_split", "type": "int", "default": 2, "description": "Minimum samples required to split"}
                        ]
                    },
                    {
                        "name": "svc",
                        "display_name": "Support Vector Classifier",
                        "description": "Support Vector Machine for classification tasks",
                        "parameters": [
                            {"name": "C", "type": "float", "default": 1.0, "description": "Regularization parameter"},
                            {"name": "kernel", "type": "string", "default": "rbf", "description": "Kernel type (rbf, linear, poly, sigmoid)"},
                            {"name": "gamma", "type": "string", "default": "scale", "description": "Kernel coefficient"}
                        ]
                    }
                ]
            },
            "unsupervised": {
                "clustering": [
                    {
                        "name": "kmeans",
                        "display_name": "K-Means Clustering",
                        "description": "Partitions data into k clusters by minimizing inertia",
                        "parameters": [
                            {"name": "n_clusters", "type": "int", "default": 3, "description": "Number of clusters"},
                            {"name": "init", "type": "string", "default": "k-means++", "description": "Method for initialization"},
                            {"name": "n_init", "type": "int", "default": 10, "description": "Number of initializations to perform"}
                        ]
                    },
                    {
                        "name": "dbscan",
                        "display_name": "DBSCAN",
                        "description": "Density-based spatial clustering of applications with noise",
                        "parameters": [
                            {"name": "eps", "type": "float", "default": 0.5, "description": "Maximum distance between samples"},
                            {"name": "min_samples", "type": "int", "default": 5, "description": "Number of samples in neighborhood"},
                            {"name": "metric", "type": "string", "default": "euclidean", "description": "Distance metric"}
                        ]
                    },
                    {
                        "name": "hierarchical",
                        "display_name": "Hierarchical Clustering",
                        "description": "Builds nested clusters by merging or splitting",
                        "parameters": [
                            {"name": "n_clusters", "type": "int", "default": 3, "description": "Number of clusters"},
                            {"name": "linkage", "type": "string", "default": "ward", "description": "Linkage criterion (ward, complete, average, single)"},
                            {"name": "affinity", "type": "string", "default": "euclidean", "description": "Distance metric"}
                        ]
                    }
                ],
                "association_rules": [
                    {
                        "name": "apriori",
                        "display_name": "Apriori Algorithm",
                        "description": "Finds frequent itemsets and association rules",
                        "parameters": [
                            {"name": "min_support", "type": "float", "default": 0.1, "description": "Minimum support threshold"},
                            {"name": "min_confidence", "type": "float", "default": 0.5, "description": "Minimum confidence threshold"},
                            {"name": "min_lift", "type": "float", "default": 1.0, "description": "Minimum lift threshold"},
                            {"name": "metric", "type": "string", "default": "lift", "description": "Metric to evaluate rules (lift, confidence, support)"}
                        ]
                    }
                ]
            }
        }
    }


@router.get("/model-compatibility")
async def check_model_compatibility(
    algorithm: str = Query(..., description="Algorithm name"),
    target_column: str = Query(None, description="Target column for supervised learning"),
    db: Session = Depends(get_db)
):
    """
    Check if an algorithm is compatible with the current dataset
    """
    try:
        # Get product data from database
        products = db.query(prod_models.Product).all()
        
        if not products:
            return {
                "success": False,
                "message": "No product data available for analysis"
            }
        
        # Convert to DataFrame
        df = pd.DataFrame([{
            "product_id": p.product_id,
            "price": p.price,
            "rating": p.rating,
            "review_count": p.review_count,
            "category": p.category,
            "brand": p.brand
        } for p in products])
        
        # Add specifications if needed and available
        if target_column and target_column.startswith("spec_"):
            spec_name = target_column.replace("spec_", "")
            specs = db.query(
                spec_models.Specification.product_id,
                spec_models.Specification.spec_value
            ).filter(spec_models.Specification.spec_name == spec_name).all()
            
            if specs:
                spec_df = pd.DataFrame([(p_id, value) for p_id, value in specs], 
                                      columns=["product_id", spec_name])
                df = df.merge(spec_df, on="product_id", how="left")
        
        # Check algorithm applicability
        applicability = check_algorithm_applicability(df, algorithm, target_column)
        
        return {
            "success": True,
            "algorithm": algorithm,
            "target_column": target_column,
            "compatible": applicability["applicable"],
            "reasons": applicability["reasons"] if not applicability["applicable"] else [],
            "dataset_info": {
                "num_samples": len(df),
                "num_features": len(df.columns) - 1,  # Exclude product_id
                "available_columns": df.columns.tolist()
            }
        }
        
    except Exception as e:
        logger.error(f"Error checking compatibility: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/data-stats")
async def get_data_statistics(db: Session = Depends(get_db)):
    """
    Get statistics about the available data for ML analysis
    """
    try:
        # Get product data
        products = db.query(prod_models.Product).all()
        
        if not products:
            return {
                "success": False,
                "message": "No product data available"
            }
        
        # Convert to DataFrame
        df = pd.DataFrame([{
            "product_id": p.product_id,
            "price": p.price,
            "rating": p.rating,
            "review_count": p.review_count,
            "category": p.category,
            "brand": p.brand
        } for p in products])
        
        # Basic statistics
        stats = {
            "num_products": len(df),
            "num_categories": df["category"].nunique(),
            "num_brands": df["brand"].nunique(),
            "price_range": [float(df["price"].min()), float(df["price"].max())],
            "rating_range": [float(df["rating"].min()), float(df["rating"].max())] if not df["rating"].isna().all() else None,
            "review_count_range": [int(df["review_count"].min()), int(df["review_count"].max())] if not df["review_count"].isna().all() else None
        }
        
        # Get specifications stats
        specs = db.query(spec_models.Specification.spec_name).distinct().all()
        spec_names = [s[0] for s in specs]
        
        # Get count for each spec
        spec_counts = {}
        for spec in spec_names:
            count = db.query(spec_models.Specification).filter(
                spec_models.Specification.spec_name == spec
            ).count()
            spec_counts[spec] = count
        
        # Top specifications by count
        top_specs = sorted(spec_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Create visualizations
        visualizations = {}
        
        # 1. Price distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(df["price"].dropna(), kde=True, ax=ax)
        ax.set_title("Price Distribution")
        ax.set_xlabel("Price")
        visualizations["price_distribution"] = fig_to_base64(fig)
        
        # 2. Rating distribution if available
        if not df["rating"].isna().all():
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(df["rating"].dropna(), kde=True, ax=ax)
            ax.set_title("Rating Distribution")
            ax.set_xlabel("Rating")
            visualizations["rating_distribution"] = fig_to_base64(fig)
        
        # 3. Category distribution
        if df["category"].nunique() <= 20:  # Only if we have a reasonable number of categories
            fig, ax = plt.subplots(figsize=(12, 8))
            category_counts = df["category"].value_counts()
            sns.barplot(x=category_counts.values, y=category_counts.index, ax=ax)
            ax.set_title("Products by Category")
            ax.set_xlabel("Count")
            visualizations["category_distribution"] = fig_to_base64(fig)
        
        # 4. Correlation matrix
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        if len(numeric_cols) >= 2:
            fig, ax = plt.subplots(figsize=(10, 8))
            corr_matrix = df[numeric_cols].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
            ax.set_title("Correlation Matrix")
            visualizations["correlation_matrix"] = fig_to_base64(fig)
        
        return {
            "success": True,
            "statistics": stats,
            "specifications": {
                "total_specs": len(spec_names),
                "top_specs": [{"name": name, "count": count} for name, count in top_specs]
            },
            "visualizations": visualizations
        }
        
    except Exception as e:
        logger.error(f"Error getting data statistics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/feature-importance")
async def calculate_feature_importance(
    target_column: str = Query(..., description="Target column for importance calculation"),
    db: Session = Depends(get_db)
):
    """
    Calculate feature importance using Random Forest
    """
    try:
        # Get product data
        products = db.query(prod_models.Product).all()

        if not products:
            raise HTTPException(status_code=404, detail="No product data available")

        # Convert to DataFrame
        df = pd.DataFrame([{
            "product_id": p.product_id,
            "price": p.price,
            "rating": p.rating,
            "review_count": p.review_count,
            "category": p.category,
            "brand": p.brand
        } for p in products])

        # Log the shape of the DataFrame
        logger.info(f"DataFrame shape: {df.shape}")

        # Validate target column exists
        if target_column not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Target column '{target_column}' not found in dataset. Available columns: {', '.join(df.columns)}"
            )

        # Separate features and target
        X = df.drop(columns=[target_column, 'product_id'], errors='ignore')
        y = df[target_column]

        # Handle missing values in target
        if y.isna().any():
            logger.warning(f"Target column '{target_column}' contains missing values. Imputing with median.")
            y = y.fillna(y.median())

        # Preprocess data
        X = preprocess_data(X)

        # Check if target is categorical or numerical
        if pd.api.types.is_numeric_dtype(y) and y.nunique() > 5:
            # Regression - use RandomForestRegressor
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model_type = "regression"
        else:
            # Classification - use RandomForestClassifier
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model_type = "classification"

        # Train the model
        model.fit(X, y)

        # Get feature importance
        importance = model.feature_importances_

        # Create DataFrame with importance values
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': importance
        })

        # Sort by importance
        importance_df = importance_df.sort_values('importance', ascending=False)

        # Generate visualization
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.barplot(x='importance', y='feature', data=importance_df, ax=ax)
        ax.set_title(f"Feature Importance for {target_column}")
        ax.set_xlabel("Importance")
        ax.set_ylabel("Feature")

        # Return results
        return {
            "success": True,
            "target_column": target_column,
            "model_type": model_type,
            "feature_importance": importance_df.to_dict('records'),
            "visualization": fig_to_base64(fig)
        }

    except ValueError as ve:
        logger.error(f"Value error in feature importance: {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
    except HTTPException as he:
        logger.error(f"HTTP exception in feature importance: {str(he)}")
        raise he
    except Exception as e:
        logger.error(f"Error in feature importance: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

