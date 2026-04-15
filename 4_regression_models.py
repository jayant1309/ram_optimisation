"""
Step 4: Baseline Regression Models

This module trains and evaluates baseline regression models for predicting dielectric
response. Models include Linear Regression, Polynomial Regression, Support Vector
Regression, and Random Forest. Generates performance metrics and comparison plots.
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
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib

sys.path.append(BASE_DIR)
from config import FEATURE_COLS, TARGET_COL, RANDOM_STATE, TEST_SIZE

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """
    Train and evaluate multiple regression models.

    Args:
        X_train (array): Training features
        X_test (array): Testing features
        y_train (array): Training targets
        y_test (array): Testing targets

    Returns:
        dict: Results dictionary with model metrics
    """
    print("[Step 4] Training baseline regression models...")

    models = {}
    results = {}

    # 1. Linear Regression
    print("  Training Linear Regression...")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    models['Linear Regression'] = lr

    y_pred_lr = lr.predict(X_test)
    results['Linear Regression'] = compute_metrics(y_test, y_pred_lr)
    results['Linear Regression']['predictions'] = y_pred_lr

    # 2. Polynomial Regression (degree 2)
    print("  Training Polynomial Regression (degree 2)...")
    poly_reg = Pipeline([
        ('poly', PolynomialFeatures(degree=2)),
        ('linear', LinearRegression())
    ])
    poly_reg.fit(X_train, y_train)
    models['Polynomial Regression'] = poly_reg

    y_pred_poly = poly_reg.predict(X_test)
    results['Polynomial Regression'] = compute_metrics(y_test, y_pred_poly)
    results['Polynomial Regression']['predictions'] = y_pred_poly

    # 3. SVR (RBF kernel)
    print("  Training SVR (RBF)...")
    svr = SVR(kernel='rbf', C=10, epsilon=0.1)
    svr.fit(X_train, y_train)
    models['SVR'] = svr

    y_pred_svr = svr.predict(X_test)
    results['SVR'] = compute_metrics(y_test, y_pred_svr)
    results['SVR']['predictions'] = y_pred_svr

    # 4. Random Forest
    print("  Training Random Forest...")
    rf = RandomForestRegressor(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    models['Random Forest'] = rf

    y_pred_rf = rf.predict(X_test)
    results['Random Forest'] = compute_metrics(y_test, y_pred_rf)
    results['Random Forest']['predictions'] = y_pred_rf
    results['Random Forest']['feature_importance'] = rf.feature_importances_

    return models, results


def compute_metrics(y_true, y_pred):
    """
    Compute regression evaluation metrics.

    Args:
        y_true (array): True target values
        y_pred (array): Predicted target values

    Returns:
        dict: Dictionary with RMSE, R², and MAE
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    return {
        'RMSE': rmse,
        'R²': r2,
        'MAE': mae
    }


def plot_actual_vs_predicted(y_test, predictions_dict):
    """
    Plot actual vs predicted values for all models.

    Args:
        y_test (array): True target values
        predictions_dict (dict): Dictionary of model predictions

    Returns:
        matplotlib.figure.Figure: Figure object
    """
    print("  Plotting Actual vs Predicted...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)
    axes = axes.flatten()

    for idx, (model_name, pred_data) in enumerate(predictions_dict.items()):
        ax = axes[idx]
        y_pred = pred_data['predictions']

        # Scatter plot
        ax.scatter(y_test, y_pred, alpha=0.6, s=20, color='steelblue')

        # Perfect prediction line
        min_val = min(np.min(y_test), np.min(y_pred))
        max_val = max(np.max(y_test), np.max(y_pred))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')

        # R² annotation
        r2 = pred_data['R²']
        ax.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax.transAxes,
                fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        ax.set_xlabel(f'Actual {TARGET_COL}', fontsize=11)
        ax.set_ylabel(f'Predicted {TARGET_COL}', fontsize=11)
        ax.set_title(model_name, fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(PLOTS_DIR, 'baseline_regression.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    if not IN_COLAB:
        plt.show()

    plt.close('all')
    return fig


def print_comparison_table(results_dict):
    """
    Print formatted comparison table of model metrics.

    Args:
        results_dict (dict): Dictionary of model results
    """
    print("\n" + "=" * 70)
    print("MODEL COMPARISON TABLE")
    print("=" * 70)

    # Create comparison dataframe
    comparison_df = pd.DataFrame({
        model_name: metrics
        for model_name, metrics in results_dict.items()
        if model_name != 'Random Forest' or (model_name == 'Random Forest' and not isinstance(metrics, dict))
    })

    # Extract metrics
    metrics_df = pd.DataFrame([
        {
            'Model': model_name,
            'RMSE': f"{metrics['RMSE']:.4f}",
            'R²': f"{metrics['R²']:.4f}",
            'MAE': f"{metrics['MAE']:.4f}"
        }
        for model_name, metrics in results_dict.items()
    ])

    print(metrics_df.to_string(index=False))
    print("=" * 70)


if __name__ == '__main__':
    import sys

    print("Loading engineered features...")
    input_path = os.path.join(DATA_DIR, 'features.csv')
    df = pd.read_csv(input_path)

    # Prepare data
    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values

    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

    try:
        # Train models
        models, results = train_and_evaluate_models(X_train, X_test, y_train, y_test)

        # Extract predictions for plotting
        predictions_dict = {name: {
            'predictions': results[name]['predictions'],
            'R²': results[name]['R²']
        } for name in results}

        # Plot predictions
        plot_actual_vs_predicted(y_test, predictions_dict)

        # Save models
        for model_name, model in models.items():
            model_path = os.path.join(DATA_DIR, f'regression_{model_name.replace(" ", "_").lower()}.pkl')
            joblib.dump(model, model_path)
            print(f"Saved {model_name} model to: {model_path}")

        # Save results
        results_to_save = {k: {mk: mv for mk, mv in v.items() if mk != 'predictions'} for k, v in results.items()}
        results_path = os.path.join(DATA_DIR, 'regression_results.pkl')
        joblib.dump(results_to_save, results_path)

        # Print comparison table
        print_comparison_table(results_to_save)

        print(f"\n[Step 4 complete] Saved results to {DATA_DIR}/regression_results.pkl")
        print(f"[Step 4 complete] Models saved to {DATA_DIR}/")

    except Exception as e:
        print(f"[ERROR] Regression training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
