"""
Step 2: Feature Engineering

This module engineers materials science features from chemical formulas using
pymatgen for composition analysis and mendeleev for element properties.
Computes weighted averages, valence electron concentration, and other relevant
features for machine learning.
"""

import os
import sys

# Detect environment
try:
    import google.colab
    IN_COLAB = True
    from google.colab import drive
    drive.mount('/content/drive', force_remount=True)
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
    try:
        from pymatgen.core import Composition
        from mendeleev import element
    except ImportError as e:
        raise

    print("Now at Step 2] Starting feature engineering...")
    df = df.copy()
    df = df.dropna(subset=[TARGET_COL])

    features = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Computing material features"):
        try:
            formula = row['formula_pretty']
            composition = Composition(formula)
            elements = list(composition.elements)
            element_counts = dict(composition.get_el_amt_dict())
            total_atoms = sum(element_counts.values())

            avg_electronegativity = 0.0
            avg_atomic_mass = 0.0
            valence_electron_conc = 0.0
            avg_ionization_energy = 0.0
            avg_atomic_radius = 0.0
            n_elements = len(elements)

            for elem in elements:
                try:
                    elem_data = element(elem.symbol)
                    count = element_counts[elem.symbol]
                    weight = count / total_atoms

                    en = getattr(elem_data, 'electronegativity_pauling', None)
                    if en is not None:
                        avg_electronegativity += en * weight

                    mass = getattr(elem_data, 'atomic_weight', None)
                    if mass is not None:
                        avg_atomic_mass += mass * weight

                    if hasattr(elem_data, 'nvalence') and callable(getattr(elem_data, 'nvalence', None)):
                        valence_electron_conc += elem_data.nvalence() * count

                    ie = elem_data.ionenergies.get(1) if hasattr(elem_data, 'ionenergies') else None
                    if ie is not None:
                        avg_ionization_energy += ie * weight

                    rad = getattr(elem_data, 'atomic_radius', None)
                    if rad is not None:
                        avg_atomic_radius += rad * weight

                except Exception as e:
                    continue

            if total_atoms > 0:
                valence_electron_conc = valence_electron_conc / total_atoms

            volume_per_atom = row['volume'] / row['nsites'] if row['volume'] and row['nsites'] else np.nan

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
                'crystal_system': row['crystal_system'],
                'e_total': row['e_total'],
                'e_electronic': row['e_electronic'],
                'e_ionic': row['e_ionic'],
                'formula_pretty': row['formula_pretty'],
            })

        except Exception as e:
            continue

    feature_df = pd.DataFrame(features)
    
    # First, we apply Label Encoding before checking for missing values
    le = LabelEncoder()
    feature_df['crystal_system_encoded'] = le.fit_transform(feature_df['crystal_system'].astype(str))
    joblib.dump(le, os.path.join(DATA_DIR, 'label_encoder.pkl'))

    missing_before = feature_df[FEATURE_COLS].isnull().sum().sum()
    if missing_before > 0:
        imputer = SimpleImputer(strategy='median')
        feature_df[FEATURE_COLS] = imputer.fit_transform(feature_df[FEATURE_COLS])
        joblib.dump(imputer, os.path.join(DATA_DIR, 'imputer.pkl'))

    final_df = feature_df[['material_id', 'formula_pretty'] + FEATURE_COLS + [TARGET_COL, 'e_electronic', 'e_ionic']]

    scaler_features = [col for col in FEATURE_COLS if col != 'crystal_system_encoded']
    scaler = StandardScaler()
    final_df[scaler_features] = scaler.fit_transform(final_df[scaler_features])
    joblib.dump(scaler, os.path.join(DATA_DIR, 'scaler.pkl'))

    output_path = os.path.join(DATA_DIR, 'features.csv')
    final_df.to_csv(output_path, index=False)
    
    print(f"\nNow at Step 2 complete] Saved engineered features to {output_path}")
    return final_df

if __name__ == '__main__':
    input_path = os.path.join(DATA_DIR, 'raw_materials.csv')
    df = pd.read_csv(input_path)
    engineered_df = compute_material_features(df)