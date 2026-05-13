"""
Step 1: Data Acquisition - Fixed Version

This module queries the Materials Project database for dielectric materials with
specified properties. It extracts raw data including dielectric constants, band gaps,
density, volume, crystal symmetry information, and material IDs.

Fixed to use correct MP API fields and endpoints based on mp-api==0.41.2
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
import sys
from tqdm import tqdm

# Import after environment setup
sys.path.append(BASE_DIR)
from config import MP_API_KEY


def acquire_materials_data():
    """
    Query Materials Project API for dielectric materials.

    Returns:
        pd.DataFrame: DataFrame containing material properties
    """
    try:
        from mp_api.client import MPRester
    except ImportError as e:
        print(f"Error importing mp_api: {e}")
        print("Make sure to install: pip install mp-api>=0.41.2")
        raise

    print("Now at Step 1] Starting data acquisition from Materials Project...")
    print(f"API Key configured: {'Yes' if MP_API_KEY != 'YOUR_KEY_HERE' else 'No (using placeholder)'}")

    # Use the summary endpoint to get both dielectric properties and band gap
    # This is more efficient than querying dielectric endpoint and filtering separately
    with MPRester(MP_API_KEY) as mpr:
        print("Querying Materials Project summary endpoint for dielectric materials...")
        docs = mpr.summary.search(
            has_props=["dielectric"],  # Filter for materials with dielectric data
            fields=[
                "material_id",
                "formula_pretty",
                "band_gap",
                "density",
                "volume",
                "nsites",
                "symmetry",
                "e_total",
                "e_ionic",
                "e_electronic"
            ]
        )

    print(f"Total materials fetched: {len(docs)}")

    # Extract relevant data
    data = []
    for doc in tqdm(docs, desc="Extracting material data"):
        try:
            # Extract dielectric properties (direct fields from API response)
            e_total = doc.e_total
            e_electronic = doc.e_electronic
            e_ionic = doc.e_ionic

            # Skip if e_total is None
            if e_total is None:
                continue

            # Extract other properties
            material_id = doc.material_id
            formula = doc.formula_pretty
            band_gap = doc.band_gap
            density = doc.density
            volume = doc.volume
            nsites = doc.nsites
            symmetry = doc.symmetry

            # Filter: only non-metals (band_gap > 0)
            if band_gap is None or band_gap <= 0:
                continue

            data.append({
                'material_id': material_id,
                'formula_pretty': formula,
                'band_gap': band_gap,
                'density': density,
                'volume': volume,
                'nsites': nsites,
                'crystal_system': symmetry.crystal_system if symmetry else None,
                'spacegroup_number': symmetry.number if symmetry else None,
                'e_total': e_total,
                'e_electronic': e_electronic,
                'e_ionic': e_ionic,
            })
        except Exception as e:
            print(f"Warning: Could not process material {doc.material_id if hasattr(doc, 'material_id') else 'unknown'}: {e}")
            continue

    df = pd.DataFrame(data)

    print(f"Filtered materials (band_gap > 0, e_total not None): {len(df)}")

    if len(df) == 0:
        raise ValueError("No materials found after filtering. Check API key and connection.")

    # Save the raw data
    output_path = os.path.join(DATA_DIR, 'raw_materials.csv')
    df.to_csv(output_path, index=False)

    print(f"Now at Step 1 complete] Saved raw materials data to {output_path}")
    print(f"Columns: {list(df.columns)}")
    print("First 5 rows:")
    print(df.head().to_string())

    return df


if __name__ == '__main__':
    try:
        df = acquire_materials_data()
    except Exception as e:
        print(f"[ERROR] Data acquisition failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
