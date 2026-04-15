"""
Step 7: Evaluation and Comparison

This module evaluates all trained models, creates comparison visualizations,
generates feature importance plots, predicts candidate materials, and produces
a comprehensive summary of the best performing model and top recommendations.
"""

import os
import sys
import importlib.util

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

# 🌟 THE FIX: Point directly to the local GitHub clone folder, NOT Google Drive
dl_path = '/content/ram_optimisation/ram_optimisation/6_deep_learning.py'

spec = importlib.util.spec_from_file_location("deep_learning", dl_path)
dl_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dl_module)

RAMNet = dl_module.RAMNet
RAMDataset = dl_module.RAMDataset


def load_all_results():
    print("[Step 7] Loading results from previous steps...")

    # Load classification results
    classification_path = os.path.join(DATA_DIR, 'classification_results.pkl')
    if os.path.exists(classification_path):
        classification_results = joblib.load(classification_path)
        print(f"Loaded classification results from {classification_path}")
    else:
        print("Warning: Classification results not found")
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
    print("\n[Step 7a] Building model comparison table...")
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
            'Accuracy': metrics.get('Accuracy', np.nan),
            'Precision': metrics.get('Precision', np.nan),
            'Recall': metrics.get('Recall', np.nan),
            'F1': metrics.get('F1', np.nan)
        })

    # Add DNN results
    if dnn_results:
        avg_metrics = dnn_results.get('final_metrics', {})
        if 'kfold_avg' in dnn_results:
             avg_metrics = dnn_results['kfold_avg']
             
        comparison_data.append({
            'Model': 'DNN',
            'Accuracy': avg_metrics.get('Accuracy', dnn_results.get('final_metrics', {}).get('Accuracy', np.nan)),
            'Precision': avg_metrics.get('Precision', dnn_results.get('final_metrics', {}).get('Precision', np.nan)),
            'Recall': avg_metrics.get('Recall', dnn_results.get('final_metrics', {}).get('Recall', np.nan)),
            'F1': avg_metrics.get('F1', dnn_results.get('final_metrics', {}).get('F1', np.nan))
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
    print("\n[Step 7b] Plotting model comparison...")

    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150)

    # Accuracy plot
    ax1 = axes
    models = comparison_df['Model'].values
    acc_values = comparison_df['Accuracy'].values

    bars1 = ax1.bar(models, acc_values, color='lightgreen', alpha=0.8, edgecolor='black')
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Accuracy Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.set_ylim(0, 1.0) 

    for bar, value in zip(bars1, acc_values):
        if pd.notnull(value):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.4f}', ha='center', va='bottom', fontsize=9)

    # F1 plot
    ax2 = axes
    f1_values = comparison_df['F1'].values

    bars2 = ax2.bar(models, f1_values, color='lightblue', alpha=0.8, edgecolor='black')
    ax2.set_ylabel('F1 Score', fontsize=12)
    ax2.set_title('F1 Score Comparison', fontsize=14, fontweight='bold')
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
    print("\n[Step 7c] Plotting feature importance...")

    rf_model_path = os.path.join(DATA_DIR, 'classification_Random Forest.pkl')
    if not os.path.exists(rf_model_path):
        rf_model_path = os.path.join(DATA_DIR, 'classification_model.pkl') 
        
    if not os.path.exists(rf_model_path):
        print("Classification Random Forest model not found. Skipping feature importance.")
        return None

    try:
        rf_model = joblib.load(rf_model_path)
        if hasattr(rf_model, 'feature_importances_'):
            importances = rf_model.feature_importances_
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
        ax.set_title('Random Forest Feature Importance', fontsize=14, fontweight='bold')

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
    print("\n[Step 7d] Predicting materials with DNN and ranking candidates...")

    features_path = os.path.join(DATA_DIR, 'features.csv')
    df = pd.read_csv(features_path)

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
    model = RAMNet(input_dim, 0.3).to(DEVICE)