"""
Configuration file for the Radar Absorbing Materials Optimization project.

This module contains all hyperparameters, feature definitions, and constants
used throughout the ML pipeline.
"""

import os

# ── API ──────────────────────────────────────────────────
# IMPORTANT: For security, it's best to set MP_API_KEY as an environment variable
# Example: export MP_API_KEY="your_key_here"
# Or create a .env file (see .env.example)
MP_API_KEY = "eZTCs8qU7fvw7p2BHdo9txw4U6BpQG4h"
# In Colab: set via the @param cell or os.environ before running

# ── Data ─────────────────────────────────────────────────
TARGET_COL = "e_total"

FEATURE_COLS = [
    "avg_electronegativity",
    "avg_atomic_mass",
    "valence_electron_conc",
    "avg_ionization_energy",
    "avg_atomic_radius",
    "band_gap",
    "density",
    "volume_per_atom",
    "n_elements",
    "crystal_system_encoded"
]

# ── ML ───────────────────────────────────────────────────
RANDOM_STATE = 22
TEST_SIZE = 0.2
N_FOLDS = 5

# ── DNN ─────────────────────────────────────────────────
EPOCHS = 200
BATCH_SIZE = 32
LEARNING_RATE = 0.001
HIDDEN_DIMS = [128, 64, 32]
DROPOUT = 0.3

# ── Device (auto GPU if available) ─────────────────────
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[config] Using device: {DEVICE}")
