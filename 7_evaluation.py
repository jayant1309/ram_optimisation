
"""
Step 7: Evaluation and Comparison

This module evaluates all trained models, creates comparison visualizations,
generates feature importance plots, predicts candidate materials, and produces
a comprehensive summary of the best performing model and top recommendations.
"""

import os
import sys
import importlib.util

# Detect environment - Adjusted for local scripts + Drive data
try:
    import google.colab
    IN_COLAB = True
    from google.colab import drive
    
    # Mount drive because data and plots are stored here
    drive.mount('/content/drive')
    DRIVE_DIR = '/content/drive/MyDrive/ram_optimisation'
    
    # The scripts themselves are stored locally in the Colab environment
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except ImportError:
    IN_COLAB = False
    DRIVE_DIR = os.path.dirname(os.path.abspath(__file__))
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Data and plots point to Google Drive
DATA_DIR = os.path.join(DRIVE_DIR, 'data')
PLOTS_DIR = os.path.join(DRIVE_DIR, 'plots')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import torch
import torch.nn as nn

# Append local script directory so we can import config.py
sys.path.append(SCRIPT_DIR)
from config import (
    FEATURE_COLS, TARGET_COL, RANDOM_STATE, TEST_SIZE,
    DEVICE, HIDDEN_DIMS, DROPOUT
)

# Robust import for model classes from the local deep learning module
try:
    # Direct import if running as script and named without numbers
    from deep_learning import RAMNet, RAMDataset
except ImportError:
    print("[INFO] Using robust module loader for deep learning classes...")
    
    # Search local script locations
    possible_paths = [
        os.path.join(SCRIPT_DIR, '6_deep_learning.py'),
        os.path.join(os.getcwd(), '6_deep_learning.py')
    ]
    
    dl_path = None
    for path in possible_paths:
        if os.path.exists(path):
            dl_path = path
            break
            
    if dl_path:
        spec = importlib.util.spec_from_file_location("deep_learning", dl_path)
        dl_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(dl_module)
        RAMNet = dl_module.RAMNet
        RAMDataset = dl_module.RAMDataset
        print(f"[INFO] Successfully loaded deep learning module from: {dl_path}")
    else:
        print("ERROR: Could not find 6_deep_learning.py.")
        print(f"Looked in: {possible_paths}")
        sys.exit(1)


def load_all_results():
    print("Now at Step 7] Loading results from previous steps...")

    # Load classification results
    classification_path = os.path.join(DATA_DIR, 'classification_results.pkl')
    if os.path.exists(classification_path):
        classification_results = joblib.load(classification_path)
        print(f"Loaded classification results from {classification_path}")
    else:
        print("Warning: Classification results not found. Did you save them in Step 5?")
        classification_results = {}

    # Load DNN results
    dnn_path = os.path.join(DATA_DIR, 'dnn_results.pkl')
    if os.path.exists(dnn_path):
        dnn_results = joblib.load(dnn_path)
        print(f"Loaded DNN results from {dnn_path}")
    else:
        print("Warning: DNN results not found")
        dnn_results = {}

    return classification_results, dnn_results


def build_comparison_table(classification_results, dnn_results):
    print("\nNow at Step 7a] Building model comparison table...")
    comparison_data = []

    model_mapping = {
        'Logistic Regression': 'LogReg',
        'SVC': 'SVC',
        'Decision Tree': 'DecisionTree',
        'Random Forest': 'RandomForest'
    }

    # Add Classical Classification Models
    for model_name, metrics in classification_results.items():
        short_name = model_mapping.get(model_name, model_name)
        comparison_data.append({
            'Model': short_name,
            'Accuracy': metrics.get('accuracy', np.nan),
            'Precision': metrics.get('precision', np.nan),
            'Recall': metrics.get('recall', np.nan),
            'F1': metrics.get('f1', np.nan)
        })

    # Add DNN results
    if dnn_results:
        avg_metrics = dnn_results.get('final_metrics', {})
        if 'kfold_avg' in dnn_results:
             avg_metrics = dnn_results['kfold_avg']
             
        comparison_data.append({
            'Model': 'DNN',
            'Accuracy': avg_metrics.get('Accuracy', np.nan),
            'Precision': avg_metrics.get('Precision', np.nan),
            'Recall': avg_metrics.get('Recall', np.nan),
            'F1': avg_metrics.get('F1', np.nan)
        })

    comparison_df = pd.DataFrame(comparison_data)

    # Format output
    display_df = comparison_df.copy()
    for col in ['Accuracy', 'Precision', 'Recall', 'F1']:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "N/A")

    print("\nModel Comparison:")
    print(display_df.to_string(index=False))

    return comparison_df


def plot_model_comparison(comparison_df):
    print("\nNow at Step 7b] Plotting model comparison...")

    plt.style.use('seaborn-v0_8-darkgrid')
    
    # BULLETPROOF FIX: Unpack the axes directly into ax1 and ax2
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=150)

    # Convert values to numeric to prevent plotting errors if data is missing
    models = comparison_df['Model'].astype(str).values
    acc_values = pd.to_numeric(comparison_df['Accuracy'], errors='coerce').values
    f1_values = pd.to_numeric(comparison_df['F1'], errors='coerce').values

    # --- Accuracy plot ---
    bars1 = ax1.bar(models, acc_values, color='lightgreen', alpha=0.8, edgecolor='black')
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Accuracy Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(len(models)))
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.set_ylim(0, 1.0) 

    for bar, value in zip(bars1, acc_values):
        if pd.notnull(value):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.4f}', ha='center', va='bottom', fontsize=9)

    # --- F1 Score plot ---
    bars2 = ax2.bar(models, f1_values, color='lightblue', alpha=0.8, edgecolor='black')
    ax2.set_ylabel('F1 Score', fontsize=12)
    ax2.set_title('F1 Score Comparison', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(len(models)))
    ax2.set_xticklabels(models, rotation=45, ha='right')
    ax2.set_ylim(0, 1.0)

    for bar, value in zip(bars2, f1_values):
        if pd.notnull(value):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.4f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    output_path = os.path.join(PLOTS_DIR, 'model_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    if not IN_COLAB:
        plt.show()

    plt.close('all')
    return fig


def plot_feature_importance():
    print("\nNow at Step 7c] Plotting feature importance...")

    # FIXED: Pointing to the Decision Tree trained in step 5 instead of the missing Random Forest
    model_path = os.path.join(DATA_DIR, 'classification_decision_tree.pkl')
    
    if not os.path.exists(model_path):
        print("Classification Decision Tree model not found. Skipping feature importance.")
        return None

    try:
        tree_model = joblib.load(model_path)
        if hasattr(tree_model, 'feature_importances_'):
            importances = tree_model.feature_importances_
        else:
            print("Model does not have feature_importances_ attribute")
            return None

        fi_df = pd.DataFrame({
            'Feature': FEATURE_COLS,
            'Importance': importances
        })
        fi_df = fi_df.sort_values('Importance', ascending=True)

        plt.style.use('seaborn-v0_8-darkgrid')
        fig, ax = plt.subplots(figsize=(10, 8), dpi=150)

        bars = ax.barh(fi_df['Feature'], fi_df['Importance'], color='steelblue', alpha=0.8)
        ax.set_xlabel('Feature Importance', fontsize=12)
        ax.set_ylabel('Feature', fontsize=12)
        ax.set_title('Decision Tree Feature Importance', fontsize=14, fontweight='bold')

        for bar, importance in zip(bars, fi_df['Importance']):
            width = bar.get_width()
            ax.text(width + 0.001, bar.get_y() + bar.get_height()/2,
                    f'{importance:.4f}', ha='left', va='center', fontsize=9)

        plt.tight_layout()
        output_path = os.path.join(PLOTS_DIR, 'feature_importance.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")

        if not IN_COLAB:
            plt.show()

        plt.close('all')
        return fig
    except Exception as e:
        print(f"Could not plot feature importance: {e}")
        return None


def predict_candidate_materials():
    print("\nNow at Step 7d] Predicting materials with DNN and ranking candidates...")

    features_path = os.path.join(DATA_DIR, 'features.csv')
    if not os.path.exists(features_path):
        print(f"Error: Features file not found at {features_path}")
        print("Please run steps 1-6 first to generate features and train models")
        return None

    try:
        df = pd.read_csv(features_path)
        print(f"Loaded {len(df)} materials from {features_path}")
    except Exception as e:
        print(f"Error loading features: {e}")
        return None

    model_path = os.path.join(DATA_DIR, 'dnn_model.pth')
    dnn_results_path = os.path.join(DATA_DIR, 'dnn_results.pkl')
    
    if not os.path.exists(model_path):
        print("DNN model not found. Cannot predict candidates.")
        return None

    threshold = np.median(df[TARGET_COL]) if TARGET_COL in df.columns else 0
    if os.path.exists(dnn_results_path):
        try:
            dnn_res = joblib.load(dnn_results_path)
            if 'classification_threshold' in dnn_res:
                threshold = dnn_res['classification_threshold']
        except:
            pass

    input_dim = len(FEATURE_COLS)
    model = RAMNet(input_dim, HIDDEN_DIMS, DROPOUT).to(DEVICE)

    # Load the trained model weights
    try:
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
        print(f"Loaded DNN model from {model_path}")
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return None

    # Make predictions on all materials
    print("\nNow at Step 7d] Scoring all materials with DNN...")
    X_tensor = torch.tensor(df[FEATURE_COLS].values, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        logits = model(X_tensor)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()

    # Attach probabilities and rank materials
    df['Probability_High_Absorber'] = probs
    df_sorted = df.sort_values('Probability_High_Absorber', ascending=False)

    # Display top 5 candidates
    print("\n" + "="*70)
    print("TOP 5 RADAR ABSORBING MATERIAL CANDIDATES")
    print("="*70)
    print(f"{'Rank':<6} {'Formula':<20} {'Probability':<15} {'Actual e_total':<15}")
    print("-"*70)

    for i in range(1, 6):
        idx = df_sorted.index[i-1]
        formula = df_sorted.loc[idx, 'formula_pretty']
        prob = df_sorted.loc[idx, 'Probability_High_Absorber'] * 100
        actual_e = df_sorted.loc[idx, TARGET_COL]

        print(f"{i:<6} {formula:<20} {prob:>10.2f}%     {actual_e:>10.4f}")

    print("="*70)
    print(f"Based on median split threshold: {threshold:.4f}")

    return df_sorted.head(5)


if __name__ == '__main__':
    print("=" * 60)
    print("STEP 7: MODEL EVALUATION AND PREDICTION")
    print("=" * 60)
    
    # 1. Load results
    classification_results, dnn_results = load_all_results()
    
    # 2. Build and display comparison table
    comparison_df = build_comparison_table(classification_results, dnn_results)
    
    # 3. Plot model comparison
    if not comparison_df.empty:
        plot_model_comparison(comparison_df)
    
    # 4. Plot feature importance
    plot_feature_importance()
    
    # 5. Predict candidates
    predict_candidate_materials()
    
    print("\nStep 7 complete Evaluation finished successfully!")