"""
Dataset Generation Module for House Price Prediction
=====================================================
Generates a realistic synthetic housing dataset with:
- Realistic feature relationships (area, bedrooms, bathrooms, location, property_age)
- Realistic target price determination with domain noise
- Missing values (to test data cleaning & imputation)
- Realistic outliers & anomalies (to test outlier detection & treatment)
"""

import os
import numpy as np
import pandas as pd

def generate_house_data(n_samples: int = 1500, random_state: int = 42) -> pd.DataFrame:
    """
    Generate synthetic house price dataset.

    Parameters:
    -----------
    n_samples : int, default=1500
        Number of housing records to generate.
    random_state : int, default=42
        Seed for reproducibility.

    Returns:
    --------
    pd.DataFrame
        Synthetic housing dataset containing raw features with noise, missing values, and outliers.
    """
    np.random.seed(random_state)

    # 1. Locations with associated pricing coefficients (economic premium)
    locations = ['Downtown', 'Lakeside', 'Midtown', 'Suburban Heights', 'Oakridge']
    location_weights = [0.25, 0.18, 0.22, 0.20, 0.15]
    loc_multiplier = {
        'Downtown': 1.40,
        'Lakeside': 1.25,
        'Midtown': 1.10,
        'Suburban Heights': 0.95,
        'Oakridge': 0.80
    }

    # Sample categorical location
    loc_choice = np.random.choice(locations, size=n_samples, p=location_weights)

    # 2. Area in square feet (log-normal distribution skewed towards typical single-family homes)
    # Median ~ 1800 sqft, range roughly 600 - 4500 sqft
    area = np.random.normal(loc=1950, scale=600, size=n_samples)
    area = np.clip(area, 600, 4800).round()

    # 3. Bedrooms (correlated with area)
    # Base bedrooms from area with Poisson variation
    base_beds = np.round((area / 650) + np.random.normal(0, 0.5, size=n_samples))
    bedrooms = np.clip(base_beds, 1, 6).astype(int)

    # 4. Bathrooms (typically 0.75 - 1.25 * bedrooms)
    base_baths = np.round(bedrooms * 0.75 + np.random.uniform(0.2, 0.8, size=n_samples))
    bathrooms = np.clip(base_baths, 1, 5).astype(int)

    # 5. Property Age in years (0 = brand new build, up to 50 years)
    property_age = np.random.exponential(scale=14, size=n_samples)
    property_age = np.clip(property_age, 0, 55).round().astype(int)

    # 6. Realistic Price calculation formula (Market Fundamentals)
    # Base price: $60,000
    # Area: ~$165/sqft base
    # Bedrooms: +$22,000 per bedroom
    # Bathrooms: +$18,000 per bathroom
    # Age depreciation: -$1,200 per year of age (with diminishing depreciation)
    # Multiplied by location premium factor + Gaussian market variance
    base_price = (
        65000 +
        area * 165 +
        bedrooms * 22000 +
        bathrooms * 19000 -
        property_age * 1250
    )

    # Apply location multipliers
    multipliers = np.array([loc_multiplier[loc] for loc in loc_choice])
    price = base_price * multipliers

    # Add realistic market noise (volatility +/- 5-8%)
    noise = np.random.normal(0, 16000, size=n_samples)
    price = np.round(price + noise, -2) # rounded to nearest 100

    # Ensure all prices stay positive
    price = np.maximum(price, 75000)

    # Create DataFrame
    df = pd.DataFrame({
        'area': area,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'location': loc_choice,
        'property_age': property_age,
        'price': price
    })

    # 7. Inject realistic missing values (MCAR / MAR) ~ 2.5% of rows
    # Area missing
    nan_area_idx = np.random.choice(n_samples, size=int(0.02 * n_samples), replace=False)
    df.loc[nan_area_idx, 'area'] = np.nan

    # Bedrooms missing
    nan_bed_idx = np.random.choice(n_samples, size=int(0.015 * n_samples), replace=False)
    df.loc[nan_bed_idx, 'bedrooms'] = np.nan

    # Bathrooms missing
    nan_bath_idx = np.random.choice(n_samples, size=int(0.015 * n_samples), replace=False)
    df.loc[nan_bath_idx, 'bathrooms'] = np.nan

    # Property age missing
    nan_age_idx = np.random.choice(n_samples, size=int(0.02 * n_samples), replace=False)
    df.loc[nan_age_idx, 'property_age'] = np.nan

    # Location missing
    nan_loc_idx = np.random.choice(n_samples, size=int(0.01 * n_samples), replace=False)
    df.loc[nan_loc_idx, 'location'] = np.nan

    # 8. Inject controlled realistic outliers
    # A few ultra-luxury mega estates (high area, high price)
    outlier_specs = [
        (0.01, {'area': 9800, 'bedrooms': 8, 'bathrooms': 7, 'property_age': 2, 'price': 2850000}),
        (0.08, {'area': 8900, 'bedrooms': 7, 'bathrooms': 6, 'property_age': 1, 'price': 2600000}),
        (0.25, {'area': 9200, 'bedrooms': 8, 'bathrooms': 8, 'property_age': 3, 'price': 2750000}),
        # Data entry anomalies / typos (e.g. extra zero on area or typo price)
        (0.40, {'area': 11500}), # typo in sqft
        (0.65, {'price': 15000}), # typo in price
        (0.85, {'area': 45}),     # typo: 45 sqft
    ]

    for pct, values in outlier_specs:
        idx = int(pct * n_samples)
        if idx < n_samples:
            for k, v in values.items():
                df.loc[idx, k] = v

    return df

def save_dataset(output_path: str = "data/house_data.csv", n_samples: int = 1500):
    """Generate and save dataset to CSV file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = generate_house_data(n_samples=n_samples)
    df.to_csv(output_path, index=False)
    print(f"Dataset successfully created with {len(df)} rows and {df.shape[1]} columns.")
    print(f"Saved to: {output_path}")
    print(f"Missing values count per column:\n{df.isnull().sum()}")
    return df

if __name__ == "__main__":
    save_dataset()
