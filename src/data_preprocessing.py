"""
Data Preprocessing and Pipeline Construction Module
===================================================
Provides robust data cleaning, missing value imputation, outlier detection,
and Scikit-Learn preprocessing pipelines for House Price Prediction.
"""

from typing import Tuple, List, Optional
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


# Core feature definitions
NUMERICAL_FEATURES = ['area', 'bedrooms', 'bathrooms', 'property_age']
CATEGORICAL_FEATURES = ['location']
TARGET_FEATURE = 'price'


def load_raw_data(filepath: str = "data/house_data.csv") -> pd.DataFrame:
    """Load raw dataset from CSV."""
    df = pd.read_csv(filepath)
    return df


def detect_outliers_iqr(df: pd.DataFrame, column: str, factor: float = 1.5) -> Tuple[float, float, pd.Series]:
    """
    Detect outliers using the Interquartile Range (IQR) technique.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe.
    column : str
        Column to analyze.
    factor : float, default=1.5
        Multiplier for IQR range.

    Returns:
    --------
    lower_bound : float
    upper_bound : float
    outlier_mask : pd.Series
        Boolean mask where True indicates an outlier.
    """
    valid_series = df[column].dropna()
    q25 = valid_series.quantile(0.25)
    q75 = valid_series.quantile(0.75)
    iqr = q75 - q25
    lower_bound = q25 - (factor * iqr)
    upper_bound = q75 + (factor * iqr)

    outlier_mask = (df[column] < lower_bound) | (df[column] > upper_bound)
    return lower_bound, upper_bound, outlier_mask


def clean_and_filter_data(
    df: pd.DataFrame,
    remove_iqr_outliers: bool = True,
    area_min: float = 300.0,
    area_max: float = 6500.0,
    price_min: float = 50000.0,
    price_max: float = 2200000.0
) -> pd.DataFrame:
    """
    Clean dataset by filtering out unrealistic domain anomalies and statistical outliers.

    Parameters:
    -----------
    df : pd.DataFrame
        Raw dataframe.
    remove_iqr_outliers : bool, default=True
        Whether to filter extreme IQR outliers on the target variable.
    area_min, area_max : float
        Domain sanity range for area (sqft).
    price_min, price_max : float
        Domain sanity range for price (USD).

    Returns:
    --------
    cleaned_df : pd.DataFrame
    """
    cleaned = df.copy()

    # 1. Domain sanity filters (removing data entry typos like 45 sqft or $15,000 price)
    initial_rows = len(cleaned)

    # Filter invalid area if not null
    cleaned = cleaned[
        cleaned['area'].isna() | ((cleaned['area'] >= area_min) & (cleaned['area'] <= area_max))
    ]

    # Filter invalid target price
    cleaned = cleaned[
        cleaned['price'].isna() | ((cleaned['price'] >= price_min) & (cleaned['price'] <= price_max))
    ]

    # Target price must never be missing in training data
    cleaned = cleaned.dropna(subset=[TARGET_FEATURE])

    # 2. IQR outlier removal for area and price
    if remove_iqr_outliers:
        _, _, price_outliers = detect_outliers_iqr(cleaned, 'price', factor=2.0)
        cleaned = cleaned[~price_outliers]

    cleaned = cleaned.reset_index(drop=True)
    filtered_rows = initial_rows - len(cleaned)
    print(f"Data Cleaning: Removed {filtered_rows} anomalous / extreme outlier rows. Cleaned dataset has {len(cleaned)} rows.")

    return cleaned


def create_preprocessor(
    num_features: Optional[List[str]] = None,
    cat_features: Optional[List[str]] = None
) -> ColumnTransformer:
    """
    Construct a Scikit-Learn ColumnTransformer pipeline for feature preprocessing.
    - Numerical: Median Imputation -> StandardScaler
    - Categorical: Most Frequent Imputation -> OneHotEncoder(drop='first', handle_unknown='ignore')

    Returns:
    --------
    ColumnTransformer
    """
    if num_features is None:
        num_features = NUMERICAL_FEATURES
    if cat_features is None:
        cat_features = CATEGORICAL_FEATURES

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_features),
            ('cat', categorical_transformer, cat_features)
        ],
        remainder='drop'
    )

    return preprocessor


def build_full_pipeline(model, preprocessor: Optional[ColumnTransformer] = None) -> Pipeline:
    """
    Package preprocessing and an ML regression estimator into an end-to-end Pipeline.
    """
    if preprocessor is None:
        preprocessor = create_preprocessor()

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
    return pipeline
