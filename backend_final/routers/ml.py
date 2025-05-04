# routers/ml.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import pandas as pd
from utils.regression import linear_regression, svr_regression
from utils.ml_helpers import encode_categorical, scale_features, train_test_split_data
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from typing import List

router = APIRouter()

# Pydantic models for request bodies
class ProductSpecs(BaseModel):
    product_id: int
    price: float
    rating: float
    review_count: int
    category: str
    brand: str
    specs: dict

class RegressionResponse(BaseModel):
    model: str
    mse: float
    r2: float

class ClusterResponse(BaseModel):
    cluster_labels: List[int]
    pca_components: List[List[float]]

# Regression Prediction - Linear Regression
@router.post("/regression/linear", response_model=RegressionResponse)
async def run_linear_regression(data: List[ProductSpecs]):
    try:
        # Convert to DataFrame
        df = pd.DataFrame([item.dict() for item in data])

        # Encode categorical columns
        df = encode_categorical(df, columns=["category", "brand"])

        # Select features and target
        df = scale_features(df, columns=["price", "review_count", "rating"])  # Example features

        X_train, X_test, y_train, y_test = train_test_split_data(df, target="price")

        # Apply Linear Regression
        model, mse, r2 = linear_regression(X_train, y_train, X_test, y_test)

        return {"model": "Linear Regression", "mse": mse, "r2": r2}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Regression Prediction - Support Vector Regression (SVR)
@router.post("/regression/svr", response_model=RegressionResponse)
async def run_svr_regression(data: List[ProductSpecs]):
    try:
        # Convert to DataFrame
        df = pd.DataFrame([item.dict() for item in data])

        # Encode categorical columns
        df = encode_categorical(df, columns=["category", "brand"])

        # Select features and target
        df = scale_features(df, columns=["price", "review_count", "rating"])  # Example features

        X_train, X_test, y_train, y_test = train_test_split_data(df, target="price")

        # Apply SVR Regression
        model, mse, r2 = svr_regression(X_train, y_train, X_test, y_test)

        return {"model": "Support Vector Regression", "mse": mse, "r2": r2}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# K-Means Clustering
@router.post("/clustering/kmeans", response_model=ClusterResponse)
async def run_kmeans_clustering(data: List[ProductSpecs]):
    try:
        # Convert to DataFrame
        df = pd.DataFrame([item.dict() for item in data])

        # Encode categorical columns
        df = encode_categorical(df, columns=["category", "brand"])

        # Scale numerical features
        df = scale_features(df, columns=["price", "review_count", "rating"])  # Example features

        # Apply KMeans clustering
        kmeans = KMeans(n_clusters=3, random_state=42)
        df["cluster"] = kmeans.fit_predict(df[["price", "review_count", "rating"]])

        # Apply PCA for dimensionality reduction
        pca = PCA(n_components=2)
        pca_components = pca.fit_transform(df[["price", "review_count", "rating"]])

        return {"cluster_labels": df["cluster"].tolist(), "pca_components": pca_components.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Random Forest Classification (Example)
@router.post("/classification/random_forest", response_model=dict)
async def run_random_forest_classification(data: List[ProductSpecs]):
    try:
        # Convert to DataFrame
        df = pd.DataFrame([item.dict() for item in data])

        # Encode categorical columns
        df = encode_categorical(df, columns=["category", "brand"])

        # Select features and target
        df = scale_features(df, columns=["price", "review_count", "rating"])  # Example features

        X_train, X_test, y_train, y_test = train_test_split_data(df, target="rating")

        # Apply Random Forest Classifier
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        # Calculate accuracy
        accuracy = (y_pred == y_test).mean()

        return {"model": "Random Forest", "accuracy": accuracy}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

