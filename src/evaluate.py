"""
Evaluation and Visualization Module for House Price Prediction
==============================================================
Provides standard regression metrics (MAE, MSE, RMSE, R2), cross-validation,
and diagnostic visualization functions.
"""

from typing import Dict, Any, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, KFold


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute comprehensive regression metrics.

    Parameters:
    -----------
    y_true : array-like of shape (n_samples,)
        Ground truth target values.
    y_pred : array-like of shape (n_samples,)
        Estimated target values.

    Returns:
    --------
    metrics : dict
        MAE, MSE, RMSE, and R2 score.
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    }


def evaluate_model_cv(model, X, y, cv: int = 5, random_state: int = 42) -> Dict[str, float]:
    """
    Perform K-Fold Cross Validation for R2 and Negative RMSE.
    """
    kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    r2_scores = cross_val_score(model, X, y, scoring='r2', cv=kf)
    neg_mse_scores = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=kf)
    rmse_scores = np.sqrt(-neg_mse_scores)

    return {
        "CV_R2_Mean": np.mean(r2_scores),
        "CV_R2_Std": np.std(r2_scores),
        "CV_RMSE_Mean": np.mean(rmse_scores),
        "CV_RMSE_Std": np.std(rmse_scores)
    }


def compare_models(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert a list of evaluation result dictionaries into a neatly sorted DataFrame.
    """
    df = pd.DataFrame(results)
    sort_col = "Test_R2" if "Test_R2" in df.columns else ("R2" if "R2" in df.columns else None)
    if sort_col:
        df = df.sort_values(by=sort_col, ascending=False).reset_index(drop=True)
    return df


def plot_actual_vs_predicted(y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "Model", save_path: str = None):
    """
    Plot Actual vs Predicted scatter plot with ideal 45-degree reference line.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.55, edgecolors='none', color='#1f77b4', label='Predictions')
    
    # Ideal line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], color='#d62728', linestyle='--', linewidth=2, label='Perfect Fit (y = x)')

    plt.title(f"{model_name}: Actual vs Predicted Prices", fontsize=14, pad=12)
    plt.xlabel("Actual Price ($)", fontsize=12)
    plt.ylabel("Predicted Price ($)", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "Model", save_path: str = None):
    """
    Plot Residual Distribution and Residuals vs Predicted values (Homoscedasticity check).
    """
    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Residual distribution
    sns.histplot(residuals, kde=True, ax=axes[0], color='#2ca02c', bins=30)
    axes[0].set_title(f"{model_name}: Residuals Distribution", fontsize=13)
    axes[0].set_xlabel("Residual (Actual - Predicted) ($)", fontsize=11)
    axes[0].set_ylabel("Count", fontsize=11)
    axes[0].axvline(0, color='red', linestyle='--', linewidth=1.5)
    axes[0].grid(True, linestyle=':', alpha=0.6)

    # 2. Residuals vs Predicted
    axes[1].scatter(y_pred, residuals, alpha=0.5, color='#9467bd')
    axes[1].axhline(0, color='red', linestyle='--', linewidth=1.5)
    axes[1].set_title(f"{model_name}: Residuals vs Fitted", fontsize=13)
    axes[1].set_xlabel("Predicted Price ($)", fontsize=11)
    axes[1].set_ylabel("Residual ($)", fontsize=11)
    axes[1].grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()
