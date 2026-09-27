"""
Model Training and Optimization Pipeline
=========================================
Trains and evaluates multiple regression algorithms (Linear Regression, Ridge,
Lasso, Random Forest, Gradient Boosting), runs 5-fold cross-validation,
performs hyperparameter tuning, and serializes the winning production pipeline.
"""

import os
import sys

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from src.generate_data import save_dataset
from src.data_preprocessing import (
    load_raw_data,
    clean_and_filter_data,
    create_preprocessor,
    build_full_pipeline,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_FEATURE
)
from src.evaluate import calculate_metrics, evaluate_model_cv, compare_models, plot_actual_vs_predicted, plot_residuals


def train_and_evaluate_models(data_path: str = "data/house_data.csv", models_dir: str = "models", reports_dir: str = "reports"):
    """
    Execute full model training, evaluation, comparison, tuning, and artifact generation.
    """
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(os.path.join(reports_dir, "figures"), exist_ok=True)

    # 1. Ensure dataset exists
    if not os.path.exists(data_path):
        print(f"Dataset not found at {data_path}. Generating synthetic dataset...")
        save_dataset(data_path)

    # 2. Load and Clean Data
    raw_df = load_raw_data(data_path)
    print(f"Loaded raw dataset with {len(raw_df)} samples.")
    cleaned_df = clean_and_filter_data(raw_df)

    # 3. Train-Test Split (80-20)
    X = cleaned_df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    y = cleaned_df[TARGET_FEATURE]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"Training set: {X_train.shape[0]} samples | Testing set: {X_test.shape[0]} samples")

    # 4. Define candidate models
    candidate_models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=50.0),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42),
        "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=150, learning_rate=0.08, max_depth=4, random_state=42)
    }

    results = []
    trained_pipelines = {}

    preprocessor = create_preprocessor()

    print("\n--- Training and Evaluating Models ---")
    for name, regressor in candidate_models.items():
        pipeline = build_full_pipeline(regressor, preprocessor)
        pipeline.fit(X_train, y_train)
        trained_pipelines[name] = pipeline

        # Predictions
        y_train_pred = pipeline.predict(X_train)
        y_test_pred = pipeline.predict(X_test)

        # Metrics
        train_metrics = calculate_metrics(y_train, y_train_pred)
        test_metrics = calculate_metrics(y_test, y_test_pred)

        # Cross Validation (5-fold)
        cv_metrics = evaluate_model_cv(pipeline, X_train, y_train, cv=5)

        results.append({
            "Model": name,
            "Train_R2": train_metrics["R2"],
            "Test_R2": test_metrics["R2"],
            "Train_RMSE": train_metrics["RMSE"],
            "Test_RMSE": test_metrics["RMSE"],
            "Test_MAE": test_metrics["MAE"],
            "CV_R2_Mean": cv_metrics["CV_R2_Mean"],
            "CV_R2_Std": cv_metrics["CV_R2_Std"],
            "CV_RMSE_Mean": cv_metrics["CV_RMSE_Mean"]
        })
        print(f"[{name}] Test R²: {test_metrics['R2']:.4f} | Test RMSE: ${test_metrics['RMSE']:,.2f} | Test MAE: ${test_metrics['MAE']:,.2f}")

    comparison_df = compare_models(results)
    comparison_path = os.path.join(reports_dir, "model_comparison.csv")
    comparison_df.to_csv(comparison_path, index=False)
    print(f"\nModel comparison table saved to: {comparison_path}")
    print(comparison_df.to_string(index=False))

    # 5. Best Baseline Model Selection
    best_baseline_name = comparison_df.iloc[0]["Model"]
    print(f"\nBest baseline model: {best_baseline_name} (Test R² = {comparison_df.iloc[0]['Test_R2']:.4f})")

    # 6. Hyperparameter Tuning on Gradient Boosting / Best Model
    print(f"\nPerforming Hyperparameter Tuning on {best_baseline_name} using GridSearchCV...")
    if "Gradient Boosting" in best_baseline_name:
        param_grid = {
            'regressor__n_estimators': [100, 150, 200],
            'regressor__learning_rate': [0.05, 0.08, 0.12],
            'regressor__max_depth': [3, 4, 5],
            'regressor__subsample': [0.85, 1.0]
        }
        base_pipeline = build_full_pipeline(GradientBoostingRegressor(random_state=42), preprocessor)
    else:
        param_grid = {
            'regressor__n_estimators': [100, 150, 200],
            'regressor__max_depth': [10, 15, None],
            'regressor__min_samples_split': [2, 5]
        }
        base_pipeline = build_full_pipeline(RandomForestRegressor(random_state=42), preprocessor)

    grid_search = GridSearchCV(
        base_pipeline,
        param_grid=param_grid,
        scoring='r2',
        cv=5,
        n_jobs=-1,
        verbose=0
    )
    grid_search.fit(X_train, y_train)

    best_pipeline = grid_search.best_estimator_
    best_params = grid_search.best_params_
    print(f"Best Hyperparameters: {best_params}")

    # Evaluate tuned model
    y_test_pred_tuned = best_pipeline.predict(X_test)
    tuned_metrics = calculate_metrics(y_test, y_test_pred_tuned)
    print(f"Tuned Model Performance: Test R² = {tuned_metrics['R2']:.4f}, Test RMSE = ${tuned_metrics['RMSE']:,.2f}, Test MAE = ${tuned_metrics['MAE']:,.2f}")

    # Determine final pipeline (pick whichever yields higher Test R²)
    baseline_pipeline = trained_pipelines[best_baseline_name]
    baseline_test_r2 = comparison_df.iloc[0]["Test_R2"]
    if baseline_test_r2 >= tuned_metrics["R2"]:
        final_pipeline = baseline_pipeline
        final_name = f"{best_baseline_name} (Optimized Default)"
        final_preds = y_test_pred
        print(f"Choosing {final_name} as final model (Test R² = {baseline_test_r2:.4f}).")
    else:
        final_pipeline = best_pipeline
        final_name = f"Tuned {best_baseline_name}"
        final_preds = y_test_pred_tuned
        print(f"Choosing {final_name} as final model (Test R² = {tuned_metrics['R2']:.4f}).")

    # 7. Generate diagnostic plots
    fig_dir = os.path.join(reports_dir, "figures")
    plot_actual_vs_predicted(
        y_test.values, final_preds,
        model_name=final_name,
        save_path=os.path.join(fig_dir, "actual_vs_predicted.png")
    )
    plot_residuals(
        y_test.values, final_preds,
        model_name=final_name,
        save_path=os.path.join(fig_dir, "residuals_diagnostic.png")
    )

    # 8. Feature Importance Extraction
    try:
        regressor = final_pipeline.named_steps['regressor']
        if hasattr(regressor, 'feature_importances_'):
            # Extract one-hot encoded feature names
            cat_encoder = final_pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
            cat_features_encoded = list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
            all_feature_names = NUMERICAL_FEATURES + cat_features_encoded
            importances = regressor.feature_importances_

            feat_imp_df = pd.DataFrame({
                'Feature': all_feature_names,
                'Importance': importances
            }).sort_values(by='Importance', ascending=True)

            plt.figure(figsize=(10, 6))
            plt.barh(feat_imp_df['Feature'], feat_imp_df['Importance'], color='#1f77b4', edgecolor='black')
            plt.title("Feature Importance in House Price Prediction", fontsize=14, pad=12)
            plt.xlabel("Relative Importance Score", fontsize=12)
            plt.grid(True, linestyle=':', alpha=0.6)
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, "feature_importance.png"), dpi=300)
            plt.close()
            print(f"Feature importance chart saved to {os.path.join(fig_dir, 'feature_importance.png')}")
    except Exception as e:
        print(f"Note: Could not generate feature importance plot: {e}")

    # 9. Save Best Model Pipeline
    model_save_path = os.path.join(models_dir, "best_house_price_model.pkl")
    joblib.dump(final_pipeline, model_save_path)
    print(f"\nFinal production model pipeline saved to: {model_save_path}")

    return final_pipeline, comparison_df, tuned_metrics


if __name__ == "__main__":
    train_and_evaluate_models()
