"""
Step 6: Deep Neural Network

This module implements a deep neural network (RAMNet) using PyTorch for
regression prediction. Includes K-Fold cross-validation, GPU acceleration,
learning rate scheduling, and comprehensive training visualization.
"""

import os
import sys

# Detect environment
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
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

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("No GPU detected. Running on CPU.")

class RAMDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        # Target must be float for BCEWithLogitsLoss
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1) 

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class RAMNet(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout):
        super(RAMNet, self).__init__()
        
        if isinstance(input_dim, (tuple, list, torch.Size)):
            input_dim = input_dim[-1]
            
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        current_dim = input_dim
        for hidden_dim in hidden_dims:
            self.layers.append(nn.Linear(current_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            self.dropouts.append(nn.Dropout(dropout))
            current_dim = hidden_dim

        # Output is 1 logit for binary classification
        self.output_layer = nn.Linear(current_dim, 1)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = self.dropouts[i](x)
        return self.output_layer(x)

def train_model_kfold(X, y, n_folds=N_FOLDS):
    print(f"[Step 6] Training Classifier with {n_folds}-Fold Cross Validation...")
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    models, train_losses_all, val_losses_all, fold_metrics = [], [], [], []

    fold_idx = 1
    for train_idx, val_idx in kf.split(X):
        print(f"Fold {fold_idx}/{n_folds}" + "="*50)
        X_train_fold, y_train_fold = X[train_idx], y[train_idx]
        X_val_fold, y_val_fold = X[val_idx], y[val_idx]

        train_loader = DataLoader(RAMDataset(X_train_fold, y_train_fold), batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(RAMDataset(X_val_fold, y_val_fold), batch_size=BATCH_SIZE, shuffle=False)

        model = RAMNet(X.shape, HIDDEN_DIMS, DROPOUT).to(DEVICE)
        
        # 🌟 CORE CLASSIFICATION FIX: Use Binary Cross Entropy with Logits
        criterion = nn.BCEWithLogitsLoss() 
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

        train_losses, val_losses = [], []
        best_val_loss = float('inf')

        for epoch in tqdm(range(EPOCHS), desc=f"Training Fold {fold_idx}"):
            model.train()
            epoch_train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(model(batch_X), batch_y)
                loss.backward()
                optimizer.step()
                epoch_train_loss += loss.item()

            avg_train_loss = epoch_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            model.eval()
            epoch_val_loss = 0.0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
                    epoch_val_loss += criterion(model(batch_X), batch_y).item()

            avg_val_loss = epoch_val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            scheduler.step(avg_val_loss)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = model.state_dict()

            if (epoch + 1) % 20 == 0:
                print(f"  Epoch [{epoch+1}/{EPOCHS}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

        models.append((model, best_model_state))
        train_losses_all.append(train_losses)
        val_losses_all.append(val_losses)

        model.load_state_dict(best_model_state)
        model.eval()

        with torch.no_grad():
            # Get logits, apply sigmoid for probabilities, and threshold at 0.5
            val_logits = model(torch.tensor(X_val_fold, dtype=torch.float32).to(DEVICE))
            val_probs = torch.sigmoid(val_logits).cpu().numpy().flatten()
            val_preds = (val_probs >= 0.5).astype(int)
            val_true = y_val_fold.flatten()

            acc = accuracy_score(val_true, val_preds)
            prec = precision_score(val_true, val_preds, zero_division=0)
            rec = recall_score(val_true, val_preds, zero_division=0)
            f1 = f1_score(val_true, val_preds, zero_division=0)

        fold_metrics.append({'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1': f1, 'best_val_loss': best_val_loss})
        print(f"Fold {fold_idx} - Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}")
        fold_idx += 1

    return models, train_losses_all, val_losses_all, fold_metrics


def train_final_model(X_train, y_train, X_test, y_test):
    print("[Step 6] Training final classifier on full dataset...")
    train_loader = DataLoader(RAMDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(RAMDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

    final_model = RAMNet(X_train.shape, HIDDEN_DIMS, DROPOUT).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(final_model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    best_test_loss = float('inf')
    
    for epoch in tqdm(range(EPOCHS), desc="Training Final Model"):
        final_model.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(final_model(batch_X), batch_y)
            loss.backward()
            optimizer.step()

        final_model.eval()
        epoch_test_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
                epoch_test_loss += criterion(final_model(batch_X), batch_y).item()

        avg_test_loss = epoch_test_loss / len(test_loader)
        scheduler.step(avg_test_loss)

        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            best_model_state = final_model.state_dict()

    final_model.load_state_dict(best_model_state)
    final_model.eval()

    with torch.no_grad():
        test_logits = final_model(torch.tensor(X_test, dtype=torch.float32).to(DEVICE))
        test_probs = torch.sigmoid(test_logits).cpu().numpy().flatten()
        test_preds = (test_probs >= 0.5).astype(int)
        test_true = y_test.flatten()

        metrics = {
            'Accuracy': accuracy_score(test_true, test_preds),
            'Precision': precision_score(test_true, test_preds, zero_division=0),
            'Recall': recall_score(test_true, test_preds, zero_division=0),
            'F1': f1_score(test_true, test_preds, zero_division=0)
        }

    return final_model, metrics, test_preds

def perform_deep_learning():
    print("=" * 60)
    print("STEP 6: DEEP NEURAL NETWORK (CLASSIFIER)")
    print("=" * 60)

    df = pd.read_csv(os.path.join(DATA_DIR, 'features.csv'))
    X = df[FEATURE_COLS].values
    y_raw = df[TARGET_COL].values
    
    # 🌟 CORE CLASSIFICATION FIX: Convert continuous target to Binary
    # We split at the median to ensure perfectly balanced classes
    threshold = np.median(y_raw)
    print(f"Creating binary targets split at median e_total: {threshold:.4f}")
    y_binary = (y_raw >= threshold).astype(float)

    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y_binary.shape}")
    print(f"Class distribution: {np.mean(y_binary)*100:.1f}% High Absorber, {(1-np.mean(y_binary))*100:.1f}% Low Absorber")

    models, train_losses_list, val_losses_list, fold_metrics = train_model_kfold(X, y_binary)

    print("\nK-Fold Cross Validation Results:")
    for i, metrics in enumerate(fold_metrics):
        print(f"Fold {i+1}: Acc={metrics['Accuracy']:.4f}, Prec={metrics['Precision']:.4f}, Rec={metrics['Recall']:.4f}, F1={metrics['F1']:.4f}")

    print("\nTraining final model on full dataset...")
    X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    
    final_model, final_metrics, final_preds = train_final_model(X_train, y_train, X_test, y_test)
    print(f"Final Test Metrics: Acc={final_metrics['Accuracy']:.4f}, Prec={final_metrics['Precision']:.4f}, Rec={final_metrics['Recall']:.4f}, F1={final_metrics['F1']:.4f}")

    torch.save(final_model.state_dict(), os.path.join(DATA_DIR, 'dnn_model.pth'))
    joblib.dump({'final_metrics': final_metrics, 'classification_threshold': threshold}, os.path.join(DATA_DIR, 'dnn_results.pkl'))
    print(f"\n[Step 6 complete] Deep learning completed successfully!")

if __name__ == '__main__':
    perform_deep_learning()