"""
Step 6: Deep Neural Network

This module implements a deep neural network (RAMNet) using PyTorch for
regression prediction. Includes K-Fold cross-validation, GPU acceleration,
learning rate scheduling, and comprehensive training visualization.
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
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import joblib
from tqdm import tqdm

sys.path.append(BASE_DIR)
from config import (
    FEATURE_COLS, TARGET_COL, RANDOM_STATE, TEST_SIZE,
    EPOCHS, BATCH_SIZE, HIDDEN_DIMS, DROPOUT, LEARNING_RATE,
    N_FOLDS, DEVICE
)

print(f"[Step 6] PyTorch version: {torch.__version__}")
print(f"[Step 6] Using device: {DEVICE}")

# Check GPU availability and memory
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
else:
    print("No GPU detected. Running on CPU.")

class RAMDataset(Dataset):
    """
    Custom Dataset class for PyTorch DataLoader.
    """
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class RAMNet(nn.Module):
    """
    Deep Neural Network for Radar Absorbing Materials prediction.

    Architecture:
    - Input layer (FEATURE_DIM)
    - Hidden layers with BatchNorm and Dropout
    - Output layer (1 neuron for regression)
    """
    def __init__(self, input_dim, hidden_dims, dropout):
        super(RAMNet, self).__init__()

        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        # Input layer
        current_dim = input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            self.layers.append(nn.Linear(current_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            self.dropouts.append(nn.Dropout(dropout))
            current_dim = hidden_dim

        # Output layer
        self.output_layer = nn.Linear(current_dim, 1)

    def forward(self, x):
        """Forward pass through the network."""
        for i, layer in enumerate(self.layers):
            x = layer(x)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = self.dropouts[i](x)

        x = self.output_layer(x)
        return x


def train_model_kfold(X, y, n_folds=N_FOLDS):
    """
    Train model using K-Fold cross-validation.

    Args:
        X (array): Feature matrix
        y (array): Target vector
        n_folds (int): Number of folds

    Returns:
        tuple: (models, train_losses, val_losses, metrics)
    """
    print(f"[Step 6] Training with {n_folds}-Fold Cross Validation...")

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    models = []
    train_losses_all = []
    val_losses_all = []
    fold_metrics = []

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    fold_idx = 1
    for train_idx, val_idx in kf.split(X):
        print(f"Fold {fold_idx}/{n_folds}" + "="*50)

        # Split data
        X_train_fold = X[train_idx]
        y_train_fold = y[train_idx]
        X_val_fold = X[val_idx]
        y_val_fold = y[val_idx]

        # Create datasets
        train_dataset = RAMDataset(X_train_fold, y_train_fold)
        val_dataset = RAMDataset(X_val_fold, y_val_fold)

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # Initialize model
        input_dim = X.shape[1]
        model = RAMNet(input_dim, HIDDEN_DIMS, DROPOUT).to(DEVICE)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

        # Training loop
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')

        for epoch in tqdm(range(EPOCHS), desc=f"Training Fold {fold_idx}"):
            # Training phase
            model.train()
            epoch_train_loss = 0.0
            batch_count = 0

            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)

                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                epoch_train_loss += loss.item()
                batch_count += 1

            avg_train_loss = epoch_train_loss / batch_count
            train_losses.append(avg_train_loss)

            # Validation phase
            model.eval()
            epoch_val_loss = 0.0
            val_batch_count = 0

            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    epoch_val_loss += loss.item()
                    val_batch_count += 1

            avg_val_loss = epoch_val_loss / val_batch_count
            val_losses.append(avg_val_loss)

            # Learning rate scheduling
            scheduler.step(avg_val_loss)

            # Early stopping check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                # Save best model for this fold
                best_model_state = model.state_dict()

            # Print progress
            if (epoch + 1) % 20 == 0:
                print(f"  Epoch [{epoch+1}/{EPOCHS}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

        # Store model and losses
        models.append((model, best_model_state))
        train_losses_all.append(train_losses)
        val_losses_all.append(val_losses)

        # Evaluate final fold performance
        model.load_state_dict(best_model_state)
        model.eval()

        with torch.no_grad():
            X_val_tensor = torch.tensor(X_val_fold, dtype=torch.float32).to(DEVICE)
            y_val_tensor = torch.tensor(y_val_fold, dtype=torch.float32).unsqueeze(1).to(DEVICE)

            val_outputs = model(X_val_tensor)
            val_predictions = val_outputs.cpu().numpy()
            val_true = y_val_fold

            rmse = np.sqrt(mean_squared_error(val_true, val_predictions))
            r2 = r2_score(val_true, val_predictions)
            mae = mean_absolute_error(val_true, val_predictions)

        fold_metrics.append({
            'RMSE': rmse,
            'R²': r2,
            'MAE': mae,
            'best_val_loss': best_val_loss
        })

        print(f"Fold {fold_idx} - RMSE: {rmse:.4f}, R²: {r2:.4f}, MAE: {mae:.4f}")

        fold_idx += 1

    return models, train_losses_all, val_losses_all, fold_metrics


def plot_kfold_loss_curves(train_losses_all, val_losses_all):
    """
    Plot training and validation loss curves for all folds.

    Args:
        train_losses_all (list): List of training loss arrays
        val_losses_all (list): List of validation loss arrays

    Returns:
        matplotlib.figure.Figure: Figure object
    """
    print("  Plotting K-Fold loss curves...")

    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(1, len(train_losses_all), figsize=(20, 4), dpi=150)
    axes = axes.flatten()

    for idx, (train_losses, val_losses) in enumerate(zip(train_losses_all, val_losses_all)):
        ax = axes[idx]

        epochs_range = range(1, len(train_losses) + 1)
        ax.plot(epochs_range, train_losses, 'b-', linewidth=2, label='Training Loss')
        ax.plot(epochs_range, val_losses, 'r-', linewidth=2, label='Validation Loss')

        ax.set_xlabel('Epoch', fontsize=10)
        ax.set_ylabel('Loss', fontsize=10)
        ax.set_title(f'Fold {idx+1}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle('K-Fold Cross Validation Loss Curves', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    output_path = os.path.join(PLOTS_DIR, 'dnn_kfold_loss_curves.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    if not IN_COLAB:
        plt.show()

    plt.close('all')
    return fig


def train_final_model(X_train, y_train, X_test, y_test):
    """
    Train final model on full training set and evaluate on test set.

    Args:
        X_train (array): Full training features
        y_train (array): Full training targets
        X_test (array): Test features
        y_test (array): Test targets

    Returns:
        tuple: (model, metrics, predictions)
    """
    print("[Step 6] Training final model on full dataset...")

    # Create datasets
    train_dataset = RAMDataset(X_train, y_train)
    test_dataset = RAMDataset(X_test, y_test)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize model
    input_dim = X_train.shape[1]
    final_model = RAMNet(input_dim, HIDDEN_DIMS, DROPOUT).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(final_model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

    # Training loop
    best_test_loss = float('inf')
    train_losses = []
    test_losses = []

    for epoch in tqdm(range(EPOCHS), desc="Training Final Model"):
        # Training phase
        final_model.train()
        epoch_train_loss = 0.0
        batch_count = 0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)

            optimizer.zero_grad()
            outputs = final_model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
            batch_count += 1

        avg_train_loss = epoch_train_loss / batch_count
        train_losses.append(avg_train_loss)

        # Test phase
        final_model.eval()
        epoch_test_loss = 0.0
        test_batch_count = 0

        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
                outputs = final_model(batch_X)
                loss = criterion(outputs, batch_y)
                epoch_test_loss += loss.item()
                test_batch_count += 1

        avg_test_loss = epoch_test_loss / test_batch_count
        test_losses.append(avg_test_loss)

        scheduler.step(avg_test_loss)

        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Train Loss: {avg_train_loss:.6f}, Test Loss: {avg_test_loss:.6f}")

        # Save best model
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            best_model_state = final_model.state_dict()

    # Load best model
    final_model.load_state_dict(best_model_state)
    final_model.eval()

    # Evaluate on test set
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
        test_outputs = final_model(X_test_tensor)
        test_predictions = test_outputs.cpu().numpy().flatten()
        test_true = y_test

        # Compute metrics
        rmse = np.sqrt(mean_squared_error(test_true, test_predictions))
        r2 = r2_score(test_true, test_predictions)
        mae = mean_absolute_error(test_true, test_predictions)

        metrics = {
            'RMSE': rmse,
            'R²': r2,
            'MAE': mae
        }

    return final_model, metrics, test_predictions


def plot_final_predictions(y_test, y_pred):
    """
    Plot actual vs predicted for final model.

    Args:
        y_test (array): True values
        y_pred (array): Predicted values

    Returns:
        matplotlib.figure.Figure: Figure object
    """
    print("  Plotting final predictions...")

    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    # Scatter plot
    ax.scatter(y_test, y_pred, alpha=0.6, s=30, color='steelblue')

    # Perfect prediction line
    min_val = min(np.min(y_test), np.min(y_pred))
    max_val = max(np.max(y_test), np.max(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')

    # Metrics
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    metrics_text = f'R² = {r2:.4f}\nRMSE = {rmse:.4f}'
    ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
            fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax.set_xlabel(f'Actual {TARGET_COL}', fontsize=12)
    ax.set_ylabel(f'Predicted {TARGET_COL}', fontsize=12)
    ax.set_title('Deep Neural Network: Actual vs Predicted', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(PLOTS_DIR, 'dnn_final_predictions.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    if not IN_COLAB:
        plt.show()

    plt.close('all')
    return fig


def perform_deep_learning():
    """
    Main function to perform deep learning with K-Fold and final training.

    Returns:
        dict: Results dictionary
    """
    print("=" * 60)
    print("STEP 6: DEEP NEURAL NETWORK")
    print("=" * 60)

    print("Loading engineered features...")
    input_path = os.path.join(DATA_DIR, 'features.csv')
    df = pd.read_csv(input_path)

    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values

    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")

    # K-Fold cross-validation
    models, train_losses_list, val_losses_list, fold_metrics = train_model_kfold(X, y)

    # Plot K-Fold loss curves
    plot_kfold_loss_curves(train_losses_list, val_losses_list)

    # Print K-Fold metrics
    print("\nK-Fold Cross Validation Results:")
    print("=" * 50)
    for i, metrics in enumerate(fold_metrics):
        print(f"Fold {i+1}: RMSE={metrics['RMSE']:.4f}, R²={metrics['R²']:.4f}, MAE={metrics['MAE']:.4f}")

    # Compute average metrics
    avg_rmse = np.mean([m['RMSE'] for m in fold_metrics])
    avg_r2 = np.mean([m['R²'] for m in fold_metrics])
    avg_mae = np.mean([m['MAE'] for m in fold_metrics])

    std_rmse = np.std([m['RMSE'] for m in fold_metrics])
    std_r2 = np.std([m['R²'] for m in fold_metrics])

    print(f"\nAverage Metrics (±Std):")
    print(f"  RMSE: {avg_rmse:.4f} ± {std_rmse:.4f}")
    print(f"  R²:   {avg_r2:.4f} ± {std_r2:.4f}")
    print(f"  MAE:  {avg_mae:.4f}")

    # Final split for final model training
    print("\nTraining final model on full dataset...")
    X_train_final, X_test_final, y_train_final, y_test_final = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    final_model, final_metrics, final_predictions = train_final_model(
        X_train_final, y_train_final, X_test_final, y_test_final
    )

    print(f"Final Test Metrics: RMSE={final_metrics['RMSE']:.4f}, R²={final_metrics['R²']:.4f}, MAE={final_metrics['MAE']:.4f}")

    # Plot final predictions
    plot_final_predictions(y_test_final, final_predictions)

    # Save final model
    model_path = os.path.join(DATA_DIR, 'dnn_model.pth')
    torch.save(final_model.state_dict(), model_path)
    print(f"Saved DNN model to: {model_path}")

    # Save results
    results = {
        'kfold_metrics': fold_metrics,
        'kfold_avg': {
            'RMSE': avg_rmse,
            'R²': avg_r2,
            'MAE': avg_mae,
            'RMSE_std': std_rmse,
            'R²_std': std_r2
        },
        'final_metrics': final_metrics
    }

    results_path = os.path.join(DATA_DIR, 'dnn_results.pkl')
    joblib.dump(results, results_path)
    print(f"Saved DNN results to: {results_path}")

    print(f"\n[Step 6 complete] Deep learning completed successfully!")

    return results


if __name__ == '__main__':
    try:
        results = perform_deep_learning()
    except Exception as e:
        print(f"[ERROR] Deep learning failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
