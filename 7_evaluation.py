"""
Step 7: Evaluation and Comparison

This module evaluates all trained models, creates comparison visualizations,
generates feature importance plots, predicts candidate materials, and produces
a comprehensive summary of the best performing model and top recommendations.
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
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import torch

sys.path.append(BASE_DIR)
from config import (
    FEATURE_COLS, TARGET_COL, RANDOM_STATE, TEST_SIZE, DEVICE
)
from 6_deep_learning import RAMNet, RAMDataset

def load_all_results():
    """
    Load results from all previous steps.

    Returns:
        tuple: (regression_results, dnn_results)
    """
    print("[Step 7] Loading results from previous steps...")

    # Load regression results
    regression_path = os.path.join(DATA_DIR, 'regression_results.pkl')
    if os.path.exists(regression_path):
        regression_results = joblib.load(regression_path)
        print(f"Loaded regression results from {regression_path}")
    else:
        print("Warning: Regression results not found")
        regression_results = {}

    # Load DNN results
    dnn_path = os.path.join(DATA_DIR, 'dnn_results.pkl')
    if os.path.exists(dnn_path):
        dnn_results = joblib.load(dnn_path)
        print(f"Loaded DNN results from {dnn_path}")
    else:
        print("Warning: DNN results not found")
        dnn_results = {}

    return regression_results, dnn_results


def build_comparison_table(regression_results, dnn_results):
    """
    Build comparison table of all models.

    Args:
        regression_results (dict): Regression model results
        dnn_results (dict): DNN results

    Returns:
        pd.DataFrame: Comparison dataframe
    """
    print("\n[Step 7a] Building model comparison table...")

    comparison_data = []

    # Add regression models
    model_mapping = {
        'Linear Regression': 'LinearReg',
        'Polynomial Regression': 'PolyReg',
        'SVR': 'SVR',
        'Random Forest': 'RandomForest'
    }

    for model_name, metrics in regression_results.items():
        short_name = model_mapping.get(model_name, model_name)
        comparison_data.append({
            'Model': short_name,
            'RMSE': metrics.get('RMSE', np.nan),
            'R²': metrics.get('R²', np.nan),
            'MAE': metrics.get('MAE', np.nan)
        })

    # Add DNN results (average from K-Fold)
    if dnn_results and 'kfold_avg' in dnn_results:
        avg_metrics = dnn_results['kfold_avg']
        comparison_data.append({
            'Model': 'DNN',
            'RMSE': avg_metrics.get('RMSE', np.nan),
            'R²': avg_metrics.get('R²', np.nan),
            'MAE': avg_metrics.get('MAE', np.nan)
        })

    comparison_df = pd.DataFrame(comparison_data)

    # Round to 4 decimals for display
    display_df = comparison_df.copy()
    for col in ['RMSE', 'R²', 'MAE']:
        display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")

    print("\nModel Comparison:")
    print(display_df.to_string(index=False))

    return comparison_df


def plot_model_comparison(comparison_df):
    """
    Plot grouped bar chart of model metrics.

    Args:
        comparison_df (pd.DataFrame): Comparison dataframe

    Returns:
        matplotlib.figure.Figure: Figure object
    """
    print("\n[Step 7b] Plotting model comparison...")

    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150)

    # RMSE plot
    ax1 = axes[0]
    models = comparison_df['Model'].values
    rmse_values = comparison_df['RMSE'].values

    bars1 = ax1.bar(models, rmse_values, color='salmon', alpha=0.8, edgecolor='black')
    ax1.set_ylabel('RMSE', fontsize=12)
    ax1.set_title('Root Mean Square Error Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticklabels(models, rotation=45, ha='right')

    # Annotate bars
    for bar, value in zip(bars1, rmse_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01 * max(rmse_values),
                f'{value:.4f}', ha='center', va='bottom', fontsize=9)

    # R² plot
    ax2 = axes[1]
    r2_values = comparison_df['R²'].values

    bars2 = ax2.bar(models, r2_values, color='lightblue', alpha=0.8, edgecolor='black')
    ax2.set_ylabel('R² Score', fontsize=12)
    ax2.set_title('R² Score Comparison', fontsize=14, fontweight='bold')
    ax2.set_xticklabels(models, rotation=45, ha='right')

    # Annotate bars
    for bar, value in zip(bars2, r2_values):
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
    """
    Extract and plot feature importance from Random Forest model.

    Returns:
        matplotlib.figure.Figure: Figure object
    """
    print("\n[Step 7c] Plotting feature importance...")

    # Load Random Forest model
    rf_model_path = os.path.join(DATA_DIR, 'regression_random_forest.pkl')
    if not os.path.exists(rf_model_path):
        print("Random Forest model not found. Skipping feature importance.")
        return None

    rf_model = joblib.load(rf_model_path)

    if hasattr(rf_model, 'feature_importances_'):
        importances = rf_model.feature_importances_
    else:
        print("Model does not have feature_importances_ attribute")
        return None

    # Create feature importance dataframe
    fi_df = pd.DataFrame({
        'Feature': FEATURE_COLS,
        'Importance': importances
    })
    fi_df = fi_df.sort_values('Importance', ascending=True)

    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)

    # Horizontal bar plot
    bars = ax.barh(fi_df['Feature'], fi_df['Importance'], color='steelblue', alpha=0.8)
    ax.set_xlabel('Feature Importance', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    ax.set_title('Random Forest Feature Importance', fontsize=14, fontweight='bold')

    # Annotate bars
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


def predict_candidate_materials():
    """
    Use DNN model to predict e_total for all materials and identify top candidates.

    Returns:
        pd.DataFrame: Top candidate materials
    """
    print("\n[Step 7d] Predicting materials with DNN and ranking candidates...")

    # Load features dataframe
    features_path = os.path.join(DATA_DIR, 'features.csv')
    df = pd.read_csv(features_path)

    # Load DNN model
    model_path = os.path.join(DATA_DIR, 'dnn_model.pth')
    if not os.path.exists(model_path):
        print("DNN model not found. Cannot predict candidates.")
        return None

    input_dim = len(FEATURE_COLS)
    model = RAMNet(input_dim, [128, 64, 32], 0.3).to(DEVICE)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Predict for all materials
    X = df[FEATURE_COLS].values
    X_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        predictions = model(X_tensor).cpu().numpy().flatten()

    # Add predictions to dataframe
    df['predicted_e_total'] = predictions

    # Sort by predicted e_total (descending for high absorbers)
    df_sorted = df.sort_values('predicted_e_total', ascending=False)

    # Get top 5 candidates
    top_5 = df_sorted.head(5)

    print("\nTop 5 Candidate Materials:")
    print("=" * 70)
    for idx, (material_id, row) in enumerate(top_5.iterrows(), 1):
        print(f"{idx}. {row['material_id']}: {row['formula_pretty']}")
        print(f"   Predicted e_total: {row['predicted_e_total']:.4f}")
        print(f"   Actual e_total: {row.get('e_total', 'N/A')}")
        print()

    return top_5


def print_final_summary(comparison_df, top_candidates):
    """
    Print the final summary with best model and top candidate.

    Args:
        comparison_df (pd.DataFrame): Model comparison dataframe
        top_candidates (pd.DataFrame): Top candidate materials
    """
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    # Find best model (highest R²)
    best_model_idx = comparison_df['R²'].idxmax()
    best_model = comparison_df.loc[best_model_idx, 'Model']
    best_r2 = comparison_df.loc[best_model_idx, 'R²']
    best_rmse = comparison_df.loc[best_model_idx, 'RMSE']

    print(f"BEST MODEL        : {best_model}")
    print(f"BEST R²           : {best_r2:.4f}")
    print(f"BEST RMSE         : {best_rmse:.4f}")

    # Get top candidate
    if top_candidates is not None and len(top_candidates) > 0:
        top_cand = top_candidates.iloc[0]
        print(f"TOP CANDIDATE     : {top_cand['formula_pretty']}")
        if 'e_total' in top_cand and not pd.isna(top_cand['e_total']):
            print(f"ACTUAL e_total    : {top_cand['e_total']:.4f}")
        print(f"PREDICTED e_total : {top_cand['predicted_e_total']:.4f}")

    print("=" * 70)


def evaluate_and_compare():
    """
    Main evaluation function that orchestrates all comparison and analysis.

    Returns:
        dict: Results dictionary
    """
    print("=" * 60)
    print("STEP 7: EVALUATION AND COMPARISON")
    print("=" * 60)

    # Load all results
    regression_results, dnn_results = load_all_results()

    if not regression_results and not dnn_results:
        print("ERROR: No results found from previous steps")
        return None

    # Build comparison table
    comparison_df = build_comparison_table(regression_results, dnn_results)

    # Plot model comparison
    plot_model_comparison(comparison_df)

    # Plot feature importance
    plot_feature_importance()

    # Predict and rank candidate materials
    top_candidates = predict_candidate_materials()

    # Print final summary
    print_final_summary(comparison_df, top_candidates)

    print(f"\n[Step 7 complete] Evaluation completed successfully!")
    print(f"All plots saved to: {PLOTS_DIR}")

    return {
        'comparison_df': comparison_df,
        'top_candidates': top_candidates
    }


if __name__ == '__main__':
    try:
        results = evaluate_and_compare()
    except Exception as e:
        print(f"[ERROR] Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
