# 🏡 House Price Prediction | Machine Learning Project

[![Python Version](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Production-Ready-brightgreen.svg)]()
[![Code Style: PEP8](https://img.shields.io/badge/Code%20Style-PEP8-black.svg)](https://pep8.org/)

An end-to-end, production-grade Machine Learning regression system developed to predict residential property prices based on physical housing characteristics, architectural age, and neighborhood locations.

---

## 📌 Table of Contents
1. [Project Overview](#-project-overview)
2. [Dataset Description & Data Dictionary](#-dataset-description--data-dictionary)
3. [System Architecture & Workflow](#-system-architecture--workflow)
4. [Exploratory Data Analysis (EDA) Insights](#-exploratory-data-analysis-eda-insights)
5. [Machine Learning Pipeline](#-machine-learning-pipeline)
6. [Model Evaluation & Comparison](#-model-evaluation--comparison)
7. [Best Model Justification](#-best-model-justification)
8. [Project Structure](#-project-structure)
9. [Installation & Setup](#-installation--setup)
10. [How to Run](#-how-to-run)
    - [1. Running the Jupyter Notebook](#1-running-the-jupyter-notebook)
    - [2. Automated Training Pipeline](#2-automated-training-pipeline)
    - [3. Interactive & CLI Price Prediction](#3-interactive--cli-price-prediction)
    - [4. Running Unit Tests](#4-running-unit-tests)
11. [Production Engineering Best Practices](#-production-engineering-best-practices)

---

## 🎯 Project Overview
Accurate real estate appraisal is vital for home buyers, sellers, mortgage lenders, and institutional investors. Traditional manual valuations can be subjective, slow, and prone to human bias.

This project delivers:
- **Statistically Grounded Data Cleaning**: Demonstrating robust imputation and Interquartile Range (IQR) outlier filtering.
- **Leakage-Free Preprocessing**: Scikit-Learn `ColumnTransformer` pipelines encapsulating numerical imputation, standard scaling, and one-hot encoding.
- **Multi-Model Benchmark**: Comparing Ordinary Least Squares (OLS), Ridge ($L_2$), Lasso ($L_1$), Random Forest Regressor, and Gradient Boosting Regressor.
- **Rigorous Evaluation**: 5-Fold Cross-Validation, RMSE, MAE, and $R^2$ scores.
- **Production Serialization**: End-to-end `.pkl` deployment artifact and interactive CLI inference engine (`predict.py`).

---

## 📊 Dataset Description & Data Dictionary
The dataset resides at `data/house_data.csv` and contains **1,500 property transactions** designed with realistic real-estate market fundamentals, injected missing values (~2% MCAR across features), and realistic anomalies to rigorously test preprocessing robustness.

| Feature Column | Data Type | Units / Range | Description |
| :--- | :--- | :--- | :--- |
| `area` | Float / Continuous | 600 – 4,800 sqft | Total interior livable square footage |
| `bedrooms` | Integer / Discrete | 1 – 6 | Number of bedrooms |
| `bathrooms` | Integer / Discrete | 1 – 5 | Number of bathrooms |
| `location` | String / Categorical | 5 distinct areas | Neighborhood (`Downtown`, `Lakeside`, `Midtown`, `Suburban Heights`, `Oakridge`) |
| `property_age` | Integer / Discrete | 0 – 55 years | Age of the property (0 = brand new construction) |
| **`price`** *(Target)* | Float / Continuous | $75,000 – $1,800,000 | Market transaction price in USD |

---

## 🛠 System Architecture & Workflow

```
┌─────────────────────────────────────────────────────────────┐
│                 1. Raw Data & Ingestion                     │
│                (data/house_data.csv)                        │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│              2. Data Cleaning & Anomaly Filter              │
│       - Domain sanity checks (min/max area & price)         │
│       - IQR outlier detection & filtering (Factor: 2.0)     │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                   3. 80-20 Train-Test Split                 │
│              (random_state=42 for reproducibility)          │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│        4. Leakage-Free Scikit-Learn Preprocessing Pipeline  │
│  - Numerical: SimpleImputer(median) ──> StandardScaler()    │
│  - Categorical: SimpleImputer(mode) ──> OneHotEncoder()     │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│        5. Multi-Model Benchmark & 5-Fold Cross Validation   │
│  - Linear, Ridge, Lasso, Random Forest, Gradient Boosting   │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│           6. Hyperparameter Optimization & Tuning           │
│        (GridSearchCV across n_estimators, depth, lr)        │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│      7. Model Persistence & Inference Artifacts             │
│   - models/best_house_price_model.pkl                       │
│   - CLI & Interactive predict.py Interface                  │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔍 Exploratory Data Analysis (EDA) Insights

1. **Area is the Dominant Factor ($r = 0.854$)**:
   - Living area displays an exceptionally strong, positive linear correlation with price.
2. **Bedrooms & Bathrooms ($r = 0.635$ & $r = 0.658$)**:
   - High positive correlation with price, while also scaling with `area`.
3. **Neighborhood Valuation Premium**:
   - **Downtown** carries the highest market multiplier (~1.40x), followed closely by scenic **Lakeside** (~1.25x).
   - **Oakridge** represents the most accessible entry-level neighborhood (~0.80x multiplier).
4. **Structural Depreciation ($r = -0.192$)**:
   - Property age exerts a consistent negative price pressure (~$1,250 depreciation per year of age).

---

## 📈 Model Evaluation & Comparison

All models were evaluated on the **same out-of-sample test split (20%, 299 samples)** and validated using **5-Fold Cross-Validation** on the training split (1,193 samples).

| Model | Test $R^2$ Score | 5-Fold CV $R^2$ (Mean ± Std) | Test RMSE ($) | Test MAE ($) | Train $R^2$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| 🥇 **Gradient Boosting Regressor** | **0.9732** | **0.9654 ± 0.0086** | **$30,809.97** | **$20,418.82** | 0.9876 |
| 🥈 **Random Forest Regressor** | **0.9606** | **0.9562 ± 0.0095** | **$37,331.24** | **$23,954.88** | 0.9933 |
| 🥉 **Lasso Regression ($L_1$)** | **0.9537** | **0.9473 ± 0.0097** | **$40,497.19** | **$29,208.61** | 0.9495 |
| 4. **Linear Regression (OLS)** | **0.9537** | **0.9473 ± 0.0098** | **$40,500.16** | **$29,214.38** | 0.9495 |
| 5. **Ridge Regression ($L_2$)** | **0.9537** | **0.9472 ± 0.0095** | **$40,508.62** | **$29,223.31** | 0.9495 |

*Summary CSV located at: `reports/model_comparison.csv`.*

---

## 🏆 Best Model Justification

**Gradient Boosting Regressor** was selected as the final production model for the following reasons:
1. **Highest Predictive Accuracy**: Achieved a test $R^2$ of **0.9732**, explaining **97.3%** of the target price variance.
2. **Lowest Error Metrics**: Minimized Test RMSE to **$30,809.97** and Test MAE to **$20,418.82** (a 24% error reduction over standard linear baselines).
3. **Generalization Stability**: 5-Fold cross-validation confirms high stability with very low variance ($0.9654 \pm 0.0086$), indicating the model is not overfitting.
4. **Non-Linear Interactions**: Successfully captures multiplicative location premiums combined with structural depreciation curves that linear models oversimplify.

---

## 📁 Project Structure

```
house price prediction ai ml/
├── .gitignore                      # Git ignore file for Python, artifacts & caches
├── pyproject.toml                  # Project configuration & pytest settings
├── requirements.txt                # Exact project dependencies
├── README.md                       # Complete project documentation & guide
├── house_price_prediction.ipynb    # Clean, fully-executed Jupyter Notebook
├── predict.py                      # Interactive & CLI model inference interface
│
├── data/
│   └── house_data.csv              # Synthetic housing transaction dataset
│
├── models/
│   └── best_house_price_model.pkl  # Serialized production pipeline
│
├── reports/
│   ├── model_comparison.csv        # Comprehensive performance metrics table
│   └── figures/
│       ├── actual_vs_predicted.png # Test set actual vs predicted scatter plot
│       ├── residuals_diagnostic.png# Histogram & homoscedasticity residual plots
│       └── feature_importance.png  # Relative feature importance ranking
│
├── src/
│   ├── __init__.py                 # Python package marker
│   ├── generate_data.py            # Dataset generator with noise & outliers
│   ├── data_preprocessing.py       # Cleaning, IQR outlier logic & pipelines
│   ├── train.py                    # Training, CV, grid search & persistence
│   └── evaluate.py                 # Evaluation metrics & diagnostic plotting
│
├── scripts/
│   └── make_notebook.py            # Programmatic notebook builder utility
│
└── tests/
    └── test_pipeline.py            # Unit tests for preprocessing & inference
```

---

## 💻 Installation & Setup

### Prerequisites
- Python 3.9+ (Python 3.10, 3.11, 3.12, 3.14 supported)
- macOS / Linux / Windows

### 1. Clone or Open the Repository
```bash
cd "house price prediction ai ml"
```

### 2. Create and Activate Virtual Environment
```bash
# Create virtual environment
python3 -m venv .venv

# Activate on macOS/Linux:
source .venv/bin/activate

# Activate on Windows:
# .venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install pytest
```

---

## 🚀 How to Run

### 1. Running the Jupyter Notebook
Launch Jupyter Lab or Notebook to view the complete step-by-step analysis:
```bash
jupyter lab house_price_prediction.ipynb
# or
jupyter notebook house_price_prediction.ipynb
```
The notebook is already fully executed with all markdown notes, summary statistics, seaborn/matplotlib figures, and model comparison tables embedded.

### 2. Automated Training Pipeline
Run the modular training pipeline script to train all models, generate diagnostic plots, and update the serialized `.pkl` model:
```bash
python3 src/train.py
```
**Console Output Example:**
```text
Loaded raw dataset with 1500 samples.
Data Cleaning: Removed 8 anomalous / extreme outlier rows. Cleaned dataset has 1492 rows.
Training set: 1193 samples | Testing set: 299 samples

--- Training and Evaluating Models ---
[Linear Regression] Test R²: 0.9537 | Test RMSE: $40,500.16 | Test MAE: $29,214.38
[Ridge Regression] Test R²: 0.9537 | Test RMSE: $40,508.62 | Test MAE: $29,223.31
[Lasso Regression] Test R²: 0.9537 | Test RMSE: $40,497.19 | Test MAE: $29,208.61
[Random Forest Regressor] Test R²: 0.9606 | Test RMSE: $37,331.24 | Test MAE: $23,954.88
[Gradient Boosting Regressor] Test R²: 0.9732 | Test RMSE: $30,809.97 | Test MAE: $20,418.82

Best baseline model: Gradient Boosting Regressor (Test R² = 0.9732)
Final production model pipeline saved to: models/best_house_price_model.pkl
```

### 3. Interactive & CLI Price Prediction

#### A. Interactive Terminal Mode
Simply run `predict.py` with no arguments to trigger the guided interactive prompt:
```bash
python3 predict.py
```
**Sample Interactive Session:**
```text
=================================================================
      HOUSE PRICE PREDICTION SYSTEM - INTERACTIVE ESTIMATOR      
=================================================================
Please enter the property details below (press Enter for default values):

1. Property Area in Sq Ft [e.g. 2100]: 2400
2. Number of Bedrooms [e.g. 3]: 4
3. Number of Bathrooms [e.g. 2]: 3

   Available Locations: Downtown, Lakeside, Midtown, Suburban Heights, Oakridge
4. Location [default: Downtown]: Lakeside
5. Property Age in Years [e.g. 5]: 4

Estimating market valuation...

=================================================================
                   VALUATION SUMMARY RESULT                      
=================================================================
  • Property Size      : 2,400 sqft
  • Layout             : 4 Beds / 3 Baths
  • Location           : Lakeside
  • Age of Property    : 4 years
-----------------------------------------------------------------
  ★ ESTIMATED PRICE    : $752,431.18 USD
  • Estimated Range    : $730,431.18 - $774,431.18 USD
=================================================================
```

#### B. Command-Line (CLI) Flag Mode
Provide arguments directly via the terminal for instant scripted valuation:
```bash
python3 predict.py --area 2100 --bedrooms 3 --bathrooms 2 --location Downtown --property_age 5
```
**Output:**
```text
Input: 2100.0 sqft, 3 beds, 2 baths, Downtown, 5 yrs
Predicted House Price: $665,912.45
```

### 4. Running Unit Tests
To verify pipeline integrity, preprocessing transformations, outlier detection, and inference:
```bash
pytest tests/test_pipeline.py -v
```
**Output:**
```text
tests/test_pipeline.py::test_data_generation PASSED               [ 20%]
tests/test_pipeline.py::test_iqr_outlier_detection PASSED         [ 40%]
tests/test_pipeline.py::test_clean_and_filter_data PASSED         [ 60%]
tests/test_pipeline.py::test_pipeline_fit_predict PASSED          [ 80%]
tests/test_pipeline.py::test_predict_function_helper PASSED       [100%]
============================== 5 passed in 1.95s ===============================
```

---

## 🛡️ Production Engineering Best Practices

- **Zero Data Leakage**: Scalers and categorical encoders are strictly fit on training data within Scikit-Learn `Pipeline` abstractions, never on the combined or test sets.
- **Robust Imputation**: Median imputation handles skewed continuous metrics; mode imputation preserves categorical validity.
- **Reproducibility**: Explicit random seeds (`random_state=42`) used throughout data generation, splits, and stochastic model training.
- **Artifact Self-Containment**: The saved `best_house_price_model.pkl` contains both the preprocessing transformers and regression estimator—allowing predictions on raw dictionary or DataFrame inputs without manual feature scaling.
- **PEP 8 Compliance**: Clean type hints, docstrings, modular organization, and clear separation of concerns.

---

## 📜 License
This project is licensed under the MIT License — free for academic, personal, and commercial portfolio use.
