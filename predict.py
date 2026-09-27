"""
House Price Prediction - Command-Line Inference Interface
=========================================================
Loads the trained production pipeline and predicts house prices
based on user-supplied or interactive specifications.
"""

import os
import sys
import argparse
import joblib
import pandas as pd

MODEL_PATH = "models/best_house_price_model.pkl"
VALID_LOCATIONS = ['Downtown', 'Lakeside', 'Midtown', 'Suburban Heights', 'Oakridge']


def load_model(path: str = MODEL_PATH):
    """Load serialized scikit-learn pipeline."""
    if not os.path.exists(path):
        print(f"Error: Model file '{path}' not found.")
        print("Please train the model first by running: python3 src/train.py")
        sys.exit(1)
    try:
        model = joblib.load(path)
        return model
    except Exception as e:
        print(f"Error loading model from '{path}': {e}")
        sys.exit(1)


def predict_price(model, area: float, bedrooms: int, bathrooms: int, location: str, property_age: int) -> float:
    """
    Format input into DataFrame and predict price using pipeline.
    """
    input_data = pd.DataFrame([{
        'area': float(area),
        'bedrooms': int(bedrooms),
        'bathrooms': int(bathrooms),
        'location': str(location),
        'property_age': int(property_age)
    }])
    
    predicted_val = model.predict(input_data)[0]
    return predicted_val


def interactive_mode(model):
    """Run an interactive prompt in terminal for the user."""
    print("=" * 65)
    print("      HOUSE PRICE PREDICTION SYSTEM - INTERACTIVE ESTIMATOR      ")
    print("=" * 65)
    print("Please enter the property details below (press Enter for default values):\n")

    # Area
    while True:
        try:
            area_input = input("1. Property Area in Sq Ft [e.g. 2100]: ").strip()
            area = float(area_input) if area_input else 2100.0
            if area <= 0:
                print("   Please enter an area greater than 0.")
                continue
            break
        except ValueError:
            print("   Invalid input. Please enter a numerical value for area.")

    # Bedrooms
    while True:
        try:
            beds_input = input("2. Number of Bedrooms [e.g. 3]: ").strip()
            bedrooms = int(beds_input) if beds_input else 3
            if bedrooms < 1:
                print("   Please enter at least 1 bedroom.")
                continue
            break
        except ValueError:
            print("   Invalid input. Please enter an integer.")

    # Bathrooms
    while True:
        try:
            baths_input = input("3. Number of Bathrooms [e.g. 2]: ").strip()
            bathrooms = int(baths_input) if baths_input else 2
            if bathrooms < 1:
                print("   Please enter at least 1 bathroom.")
                continue
            break
        except ValueError:
            print("   Invalid input. Please enter an integer.")

    # Location
    print(f"\n   Available Locations: {', '.join(VALID_LOCATIONS)}")
    while True:
        loc_input = input("4. Location [default: Downtown]: ").strip()
        location = loc_input if loc_input else "Downtown"
        # Match case-insensitively
        matched_loc = next((loc for loc in VALID_LOCATIONS if loc.lower() == location.lower()), None)
        if matched_loc:
            location = matched_loc
            break
        else:
            print(f"   Unknown location '{location}'. Please choose from: {', '.join(VALID_LOCATIONS)}")

    # Property Age
    while True:
        try:
            age_input = input("5. Property Age in Years [e.g. 5]: ").strip()
            property_age = int(age_input) if age_input else 5
            if property_age < 0:
                print("   Property age cannot be negative.")
                continue
            break
        except ValueError:
            print("   Invalid input. Please enter an integer for property age.")

    print("\nEstimating market valuation...")
    price = predict_price(model, area, bedrooms, bathrooms, location, property_age)

    # Estimate rough 90% confidence interval based on test RMSE (~$20k)
    lower_bound = max(0, price - 22000)
    upper_bound = price + 22000

    print("\n" + "=" * 65)
    print("                   VALUATION SUMMARY RESULT                      ")
    print("=" * 65)
    print(f"  • Property Size      : {area:,.0f} sqft")
    print(f"  • Layout             : {bedrooms} Beds / {bathrooms} Baths")
    print(f"  • Location           : {location}")
    print(f"  • Age of Property    : {property_age} years")
    print("-" * 65)
    print(f"  ★ ESTIMATED PRICE    : ${price:,.2f} USD")
    print(f"  • Estimated Range    : ${lower_bound:,.2f} - ${upper_bound:,.2f} USD")
    print("=" * 65 + "\n")


def parse_arguments():
    """Parse command line flags."""
    parser = argparse.ArgumentParser(description="Estimate house prices using trained ML model.")
    parser.add_argument("--area", type=float, help="Property area in square feet")
    parser.add_argument("--bedrooms", type=int, help="Number of bedrooms")
    parser.add_argument("--bathrooms", type=int, help="Number of bathrooms")
    parser.add_argument("--location", type=str, choices=VALID_LOCATIONS, help=f"Location ({', '.join(VALID_LOCATIONS)})")
    parser.add_argument("--property_age", type=int, help="Age of property in years")
    parser.add_argument("--model-path", type=str, default=MODEL_PATH, help="Path to saved model pipeline (.pkl)")
    return parser.parse_args()


def main():
    args = parse_arguments()
    model = load_model(args.model_path)

    # If any CLI arguments were supplied, run in CLI mode; otherwise run interactive mode
    if any([args.area, args.bedrooms, args.bathrooms, args.location, args.property_age]):
        area = args.area if args.area is not None else 2000.0
        bedrooms = args.bedrooms if args.bedrooms is not None else 3
        bathrooms = args.bathrooms if args.bathrooms is not None else 2
        location = args.location if args.location is not None else "Downtown"
        property_age = args.property_age if args.property_age is not None else 5

        price = predict_price(model, area, bedrooms, bathrooms, location, property_age)
        print(f"Input: {area} sqft, {bedrooms} beds, {bathrooms} baths, {location}, {property_age} yrs")
        print(f"Predicted House Price: ${price:,.2f}")
    else:
        interactive_mode(model)


if __name__ == "__main__":
    main()
