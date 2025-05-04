# utils/ml_helpers.py

import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from typing import Tuple

# Function to encode categorical variables into numeric format (Label Encoding)
def encode_categorical(data: pd.DataFrame, columns: list) -> pd.DataFrame:
    label_encoder = LabelEncoder()
    for col in columns:
        data[col] = label_encoder.fit_transform(data[col])
    return data

# Function to scale the features for ML models
def scale_features(data: pd.DataFrame, columns: list) -> pd.DataFrame:
    scaler = StandardScaler()
    data[columns] = scaler.fit_transform(data[columns])
    return data

# Function to split the dataset into training and testing sets
def train_test_split_data(data: pd.DataFrame, target: str, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X = data.drop(target, axis=1)
    y = data[target]
    return train_test_split(X, y, test_size=test_size, random_state=42)

