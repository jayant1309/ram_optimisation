"""
Step 5: Classification Models

This module performs binary classification to predict whether a material is a
high or low absorber based on the median e_total value. Trains Logistic Regression,
SVM, and Decision Tree classifiers with evaluation metrics and confusion matrices.
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
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import joblib

sys.path.append(BASE_DIR)
from config import FEATURE_COLS, TARGET_COL, RANDOM_STATE, TEST_SIZE

def create_binary_target(df):
    """
    Create binary target variable based on median e_total.

    Args:
        df (pd.DataFrame): DataFrame with e_total column

    Returns:
        pd.Series: Binary target series (0=low, 1=high)
    """
    median_e_total = df[TARGET_COL].median()
    high_absorber = (df[TARGET_COL] > median_e_total).astype(int)
    print(f"Binary target created with threshold (median): {median_e_total:.4f}")
    print(f"Class distribution: {high_absorber.value_counts().to_dict()}")
    return high_absorber


def train_classification_models(X_train, X_test, y_train, y_test):
    """
    Train multiple classification models and evaluate performance.

    Args:
        X_train (array): Training features
        X_test (array): Testing features
        y_train (array): Training targets
        y_test (array): Testing targets

    Returns:
        tuple: (trained_models, results_dict)
    """
    print("Starting Step 5: Training classification models...")

    models = {}
    results = {}

    # 1. Logistic Regression
    print("  Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    lr.fit(X_train, y_train)
    models['Logistic Regression'] = lr

    y_pred_lr = lr.predict(X_test)
    results['Logistic Regression'] = compute_classification_metrics(y_test, y_pred_lr)
    results['Logistic Regression']['predictions'] = y_pred_lr

    # 2. Support Vector Machine
    print("  Training SVC...")
    svc = SVC(kernel='rbf', C=10, probability=True, random_state=RANDOM_STATE)
    svc.fit(X_train, y_train)
    models['SVC'] = svc

    y_pred_svc = svc.predict(X_test)
    results['SVC'] = compute_classification_metrics(y_test, y_pred_svc)
    results['SVC']['predictions'] = y_pred_svc

    # 3. Decision Tree
    print("  Training Decision Tree...")
    dt = DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE)
    dt.fit(X_train, y_train)
    models['Decision Tree'] = dt

    y_pred_dt = dt.predict(X_test)
    results['Decision Tree'] = compute_classification_metrics(y_test, y_pred_dt)
    results['Decision Tree']['predictions'] = y_pred_dt

    return models, results


def compute_classification_metrics(y_true, y_pred):
    """
    Compute classification evaluation metrics.

    Args:
        y_true (array): True labels
        y_pred (array): Predicted labels

    Returns:
        dict: Dictionary with accuracy, precision, recall, f1
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def plot_confusion_matrix(y_true, y_pred, model_name, class_names=['Low Absorber', 'High Absorber']):
    """
    Plot confusion matrix for a classification model.

    Args:
        y_true (array): True labels
        y_pred (array): Predicted labels
        model_name (str): Name of the model
        class_names (list): Names of the classes

    Returns:
        matplotlib.figure.Figure: Figure object
    """
    print(f"  Plotting confusion matrix for {model_name}...")

    cm = confusion_matrix(y_true, y_pred)

    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)

    # Plot heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, cbar_kws={'shrink': 0.8})

    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title(f'Confusion Matrix: {model_name}', fontsize=14, fontweight='bold')

    # Add metrics to plot
    metrics = compute_classification_metrics(y_true, y_pred)
    metrics_text = f"Accuracy: {metrics['accuracy']:.4f}\n"
    metrics_text += f"Precision: {metrics['precision']:.4f}\n"
    metrics_text += f"Recall: {metrics['recall']:.4f}\n"
    metrics_text += f"F1 Score: {metrics['f1']:.4f}"

    ax.text(1.15, 0.5, metrics_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    output_path = os.path.join(PLOTS_DIR, f'classification_confusion_{model_name.replace(" ", "_").lower()}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    if not IN_COLAB:
        plt.show()

    plt.close('all')
    return fig


def print_classification_reports(y_true_dict, results_dict):
    """
    Print classification reports for all models.

    Args:
        y_true_dict (dict): Dictionary of true labels for each model
        results_dict (dict): Dictionary of results with predictions
    """
    print("\n" + "=" * 70)
    print("CLASSIFICATION REPORTS")
    print("=" * 70)

    for model_name in results_dict:
        print(f"\nModel: {model_name}")
        print("-" * 50)
        y_true = y_true_dict[model_name]
        y_pred = results_dict[model_name]['predictions']
        print(classification_report(y_true, y_pred, target_names=['Low Absorber', 'High Absorber']))


def print_comparison_table(results_dict):
    """
    Print formatted comparison table of model metrics.

    Args:
        results_dict (dict): Dictionary of model results
    """
    print("\n" + "=" * 70)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 70)

    # Create comparison dataframe
    comparison_data = []
    for model_name, metrics in results_dict.items():
        comparison_data.append({
            'Model': model_name,
            'Accuracy': f"{metrics['accuracy']:.4f}",
            'Precision': f"{metrics['precision']:.4f}",
            'Recall': f"{metrics['recall']:.4f}",
            'F1': f"{metrics['f1']:.4f}"
        })

    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))
    print("=" * 70)


if __name__ == '__main__':
    import sys

    print("Loading engineered features...")
    input_path = os.path.join(DATA_DIR, 'features.csv')
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} materials")

    # Create binary target
    y_binary = create_binary_target(df)
    df['high_absorber'] = y_binary

    # Prepare data
    X = df[FEATURE_COLS].values
    y = y_binary.values

    print(f"Features shape: {X.shape}")
    print(f"Binary target shape: {y.shape}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    print(f"Training class distribution: {pd.Series(y_train).value_counts().to_dict()}")

    try:
        # Train models
        models, results = train_classification_models(X_train, X_test, y_train, y_test)

        # Plot confusion matrices
        plt.style.use('seaborn-v0_8-darkgrid')
        for model_name, model in models.items():
            y_pred = results[model_name]['predictions']
            plot_confusion_matrix(y_test, y_pred, model_name)

        # Save classification models
        for model_name, model in models.items():
            model_path = os.path.join(DATA_DIR, f'classification_{model_name.replace(" ", "_").lower()}.pkl')
            joblib.dump(model, model_path)
            print(f"Saved {model_name} model to: {model_path}")

        # Print classification reports and comparison
        y_true_dict = {model_name: y_test for model_name in models}
        print_classification_reports(y_true_dict, results)
        print_comparison_table(results)

        joblib.dump(results, os.path.join(DATA_DIR, 'classification_results.pkl'))

        print(f"\nStep 5 complete Classification completed successfully!")
        print(f"Models and results saved in {DATA_DIR}/")

    except Exception as e:
        print(f"[ERROR] Classification failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
