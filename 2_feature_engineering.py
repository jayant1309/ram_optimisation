"""
Step 2: Feature Engineering

This module engineers materials science features from chemical formulas using
pymatgen for composition analysis and mendeleev for element properties.
Computes weighted averages, valence electron concentration, and other relevant
features for machine learning.
"""

import os
import sys

# Detect environment - must be at top of every file
try:
    import google.colab
    IN_COLAB = True
    from google.colab import drive
    drive.mount('/content/drive')
    BASE_DIR = '/content/drive/MyDrive/ram_optimisation'
except ImportError:
    IN_COLAB = False
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, 'data')
PLOTS_DIR = os.path.join(BASE_DIR, 'plots')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import joblib

sys.path.append(BASE_DIR)
from config import FEATURE_COLS, TARGET_COL

def compute_material_features(df):
    """
    Compute materials science features from chemical formulas.

    Args:
        df (pd.DataFrame): DataFrame with formula_pretty column

    Returns:
        pd.DataFrame: DataFrame with engineered features
    """
    try:
        from pymatgen.core import Composition
        from mendeleev import element
    except ImportError as e:
        print(f"Error importing required libraries: {e}")
        print("Make sure to install: pip install pymatgen>=2024.11.13 mendeleev>=0.19.0")
        raise

    print("[Step 2] Starting feature engineering...")

    # Create a copy to avoid modifying original
    df = df.copy()
    initial_rows = len(df)
    print(f"Initial rows: {initial_rows}")

    # Drop rows where e_total is NaN
    df = df.dropna(subset=[TARGET_COL])
    print(f"Rows after dropping NaN e_total: {len(df)} (dropped {initial_rows - len(df)})")

    # Extract composition and compute features for each material
    features = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Computing material features"):
        try:
            formula = row['formula_pretty']
            composition = Composition(formula)

            # Get element information
            elements = list(composition.elements)
            element_counts = dict(composition.get_el_amt_dict())
            total_atoms = sum(element_counts.values())

            # Initialize feature values
            avg_electronegativity = 0.0
            avg_atomic_mass = 0.0
            valence_electron_conc = 0.0
            avg_ionization_energy = 0.0
            avg_atomic_radius = 0.0
            n_elements = len(elements)

            # Compute weighted averages
            for elem in elements:
                try:
                    elem_data = element(elem.symbol)
                    count = element_counts[elem.symbol]
                    weight = count / total_atoms

                    # Weighted mean electronegativity (Pauling scale)
                    avg_electronegativity += elem_data.electronegativity_pauling * weight

                    # Weighted mean atomic mass
                    avg_atomic_mass += elem_data.atomic_weight * weight

                    # Valence electron concentration (sum of valence electrons)
                    if hasattr(elem_data, 'nvalence'):
                        valence_electron_conc += elem_data.nvalence() * count

                    # Mean ionization energy
                    avg_ionization_energy += elem_data.ionization_energy * weight

                    # Mean atomic radius
                    if elem_data.atomic_radius:
                        avg_atomic_radius += elem_data.atomic_radius * weight

                except Exception as e:
                    print(f"Warning: Could not fetch data for element {elem.symbol}: {e}")
                    continue

            # Normalize valence electron concentration per atom
            if total_atoms > 0:
                valence_electron_conc = valence_electron_conc / total_atoms

            # Additional features from raw data
            volume_per_atom = row['volume'] / row['nsites'] if row['volume'] and row['nsites'] else np.nan
            crystal_system = row['crystal_system']

            features.append({
                'material_id': row['material_id'],
                'avg_electronegativity': avg_electronegativity,
                'avg_atomic_mass': avg_atomic_mass,
                'valence_electron_conc': valence_electron_conc,
                'avg_ionization_energy': avg_ionization_energy,
                'avg_atomic_radius': avg_atomic_radius,
                'band_gap': row['band_gap'],
                'density': row['density'],
                'volume_per_atom': volume_per_atom,
                'n_elements': n_elements,
                'crystal_system': crystal_system,
                'e_total': row['e_total'],
                'e_electronic': row['e_electronic'],
                'e_ionic': row['e_ionic'],
                'formula_pretty': row['formula_pretty'],
            })

        except Exception as e:
            print(f"Warning: Could not process material at index {idx}: {e}")
            continue

    # Create DataFrame from features
    feature_df = pd.DataFrame(features)
    print(f"Features computed for {len(feature_df)} materials")

    # Handle missing values in features (not target)
    missing_before = feature_df[FEATURE_COLS].isnull().sum().sum()
    print(f"Missing values before imputation: {missing_before}")

    if missing_before > 0:
        # Use median imputation for numeric features
        imputer = SimpleImputer(strategy='median')
        feature_array = imputer.fit_transform(feature_df[FEATURE_COLS])

        # Convert back to DataFrame
        feature_df[FEATURE_COLS] = feature_array

        missing_after = feature_df[FEATURE_COLS].isnull().sum().sum()
        print(f"Missing values after imputation: {missing_after}")

        # Save imputer
        imputer_path = os.path.join(DATA_DIR, 'imputer.pkl')
        joblib.dump(imputer, imputer_path)
        print(f"Saved imputer to {imputer_path}")
    else:
        print("No missing values found, no imputation needed")

    # Encode crystal system
    le = LabelEncoder()
    feature_df['crystal_system_encoded'] = le.fit_transform(
        feature_df['crystal_system'].astype(str)
    )

    # Save the label encoder
    encoder_path = os.path.join(DATA_DIR, 'label_encoder.pkl')
    joblib.dump(le, encoder_path)
    print(f"Saved label encoder to {encoder_path}")

    # Select final features
    final_df = feature_df[['material_id', 'formula_pretty'] + FEATURE_COLS + [TARGET_COL, 'e_electronic', 'e_ionic']]

    # Apply StandardScaler to numerical features only (excluding material_id, formula, and targets)
    scaler_features = [col for col in FEATURE_COLS if col != 'crystal_system_encoded']
    scaler = StandardScaler()
    final_df[scaler_features] = scaler.fit_transform(final_df[scaler_features])

    # Save the scaler
    scaler_path = os.path.join(DATA_DIR, 'scaler.pkl')
    joblib.dump(scaler, scaler_path)
    print(f"Saved scaler to {scaler_path}")

    # Save the final feature dataframe
    output_path = os.path.join(DATA_DIR, 'features.csv')
    final_df.to_csv(output_path, index=False)

    print(f"[Step 2 complete] Saved engineered features to {output_path}")
    print(f"Final shape: {final_df.shape}")
    print(f"Final columns: {list(final_df.columns)}")
    print("\nFeature statistics:")
    print(final_df[FEATURE_COLS].describe().round(3))

    return final_df


if __name__ == '__main__':
    import sys

    print("Loading raw materials data...")
    input_path = os.path.join(DATA_DIR, 'raw_materials.csv')
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} materials from {input_path}")

    try:
        engineered_df = compute_material_features(df)
    except Exception as e:
        print(f"[ERROR] Feature engineering failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print(f"\n[Step 2 complete] Features engineered and saved successfully!")
