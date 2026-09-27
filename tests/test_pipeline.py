"""
Unit Tests for House Price Prediction Pipeline
===============================================
Tests data generation, preprocessing, outlier filtering, and model inference.
"""

import os
import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from src.generate_data import generate_house_data
from src.data_preprocessing import (
    clean_and_filter_data,
    detect_outliers_iqr,
    create_preprocessor,
    build_full_pipeline,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_FEATURE
)
from predict import predict_price


def test_data_generation():
    """Test that synthetic data generator creates required features and rows."""
    df = generate_house_data(n_samples=200, random_state=42)
    assert len(df) == 200
    expected_cols = {'area', 'bedrooms', 'bathrooms', 'location', 'property_age', 'price'}
    assert expected_cols.issubset(set(df.columns))

    # Test presence of missing values
    assert df.isnull().sum().sum() > 0, "Synthetic dataset should contain missing values for cleaning"


def test_iqr_outlier_detection():
    """Test IQR outlier detection logic."""
    df = pd.DataFrame({'value': [10, 12, 11, 13, 12, 100, 11, -50]})
    lower, upper, mask = detect_outliers_iqr(df, 'value', factor=1.5)
    assert mask.sum() >= 2
    assert mask.iloc[5] == True  # 100 is an outlier
    assert mask.iloc[7] == True  # -50 is an outlier


def test_clean_and_filter_data():
    """Test data cleaning removes anomalies."""
    raw_df = generate_house_data(n_samples=300, random_state=42)
    cleaned_df = clean_and_filter_data(raw_df)
    assert len(cleaned_df) <= len(raw_df)
    assert cleaned_df[TARGET_FEATURE].isnull().sum() == 0
    assert (cleaned_df['area'].dropna() >= 300).all()
    assert (cleaned_df['price'].dropna() >= 50000).all()


def test_pipeline_fit_predict():
    """Test that full pipeline handles missing values, categorical encoding, and inference."""
    raw_df = generate_house_data(n_samples=250, random_state=42)
    cleaned = clean_and_filter_data(raw_df)

    X = cleaned[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    y = cleaned[TARGET_FEATURE]

    pipeline = build_full_pipeline(Ridge(alpha=1.0))
    pipeline.fit(X, y)

    # Test prediction on unseen input
    sample = pd.DataFrame([{
        'area': 2100.0,
        'bedrooms': 3,
        'bathrooms': 2,
        'location': 'Downtown',
        'property_age': 5
    }])
    pred = pipeline.predict(sample)
    assert len(pred) == 1
    assert pred[0] > 0, "Predicted price should be positive"


def test_predict_function_helper():
    """Test inference helper function."""
    raw_df = generate_house_data(n_samples=250, random_state=42)
    cleaned = clean_and_filter_data(raw_df)
    pipeline = build_full_pipeline(Ridge())
    pipeline.fit(cleaned[NUMERICAL_FEATURES + CATEGORICAL_FEATURES], cleaned[TARGET_FEATURE])

    val = predict_price(pipeline, 1800, 3, 2, 'Lakeside', 10)
    assert isinstance(val, (float, np.floating))
    assert val > 100000
