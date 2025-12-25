import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import KFold
from torch.utils.data import TensorDataset, DataLoader
from typing import Optional

class SklearnStyleMLP(nn.Module):
    """
    Replicates the structure of sklearn's MLPClassifier:
    hidden_layer_sizes=(10*dim, 10*dim), activation='relu'.
    """
    def __init__(self, input_dim):
        super().__init__()
        hidden_dim = 10 * input_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1) 
        )
        # Trick 2: Explicitly align weight initialization with sklearn
        # Sklearn uses Glorot (Xavier) Uniform for weights and Zeros for bias.
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # Xavier Uniform
            nn.init.xavier_uniform_(m.weight)
            # Bias initialized to 0
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.net(x)

def c2st(
    X: torch.Tensor,
    Y: torch.Tensor,
    seed: int = 1,
    n_folds: int = 5,
    scoring: str = "accuracy",
    z_score: bool = True,
    noise_scale: Optional[float] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> torch.Tensor:
    """
    Classifier Two-Sample Test (C2ST) implemented in pure PyTorch.
    
    This implementation mimics sklearn.neural_network.MLPClassifier behavior 
    (including L2 regularization, initialization, and early stopping) 
    but runs entirely on GPU for acceleration.
    """
    
    # 1. Move data to device
    X = X.to(device)
    Y = Y.to(device)

    # 2. Preprocessing: Z-Score Normalization
    if z_score:
        X_mean = torch.mean(X, dim=0, keepdim=True)
        X_std = torch.std(X, dim=0, keepdim=True)
        # Avoid division by zero
        X_std[X_std == 0] = 1.0 
        X = (X - X_mean) / X_std
        Y = (Y - X_mean) / X_std

    # 3. Preprocessing: Noise Injection
    if noise_scale is not None:
        X += noise_scale * torch.randn_like(X)
        Y += noise_scale * torch.randn_like(Y)

    # 4. Prepare dataset and labels
    # Concatenate data: P (X) -> Label 0, Q (Y) -> Label 1
    data = torch.cat((X, Y), dim=0)
    target = torch.cat((
        torch.zeros(X.shape[0], device=device),
        torch.ones(Y.shape[0], device=device)
    ), dim=0).unsqueeze(1) # Shape: (N, 1)

    ndim = X.shape[1]
    
    # 5. K-Fold Cross-Validation setup
    # KFold generates indices on CPU, but slicing happens on GPU
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    indices = np.arange(len(data))
    fold_scores = []

    # --- Sklearn Alignment Hyperparameters ---
    
    batch_size = min(200, len(data)) 
    weight_decay = 1e-4 
    
    lr = 1e-3
    max_epochs = 1000
    n_iter_no_change = 10
    tol = 1e-4

    for train_idx, test_idx in kf.split(indices):
        # Slice data tensors directly on GPU
        X_train = data[train_idx]
        y_train = target[train_idx]
        X_test = data[test_idx]
        y_test = target[test_idx]

        train_ds = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        # Reset model for each fold
        model = SklearnStyleMLP(ndim).to(device)
        
        # Initialize optimizer with L2 regularization
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.BCEWithLogitsLoss()

        # Variables for Early Stopping
        best_loss = float('inf')
        no_improve_count = 0
        
        # --- Training Loop ---
        model.train()
        for epoch in range(max_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for x_batch, y_batch in train_loader:
                optimizer.zero_grad()
                logits = model(x_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches

            # --- Early Stopping Logic (based on Training Loss) ---
            # Mimics sklearn's check: loss must improve by at least 'tol'
            if avg_loss < (best_loss - tol):
                best_loss = avg_loss
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            if no_improve_count >= n_iter_no_change:
                # Stop training if no improvement for n_iter_no_change epochs
                break
        
        # --- Evaluation ---
        model.eval()
        with torch.no_grad():
            logits_test = model(X_test)
            probs_test = torch.sigmoid(logits_test)
            
            if scoring == "accuracy":
                preds = (probs_test > 0.5).float()
                acc = (preds == y_test).float().mean().item()
                fold_scores.append(acc)
            elif scoring == "roc_auc":
                # Fallback to sklearn for AUC calculation if needed, 
                # or implement a pure torch AUC function.
                try:
                    from sklearn.metrics import roc_auc_score
                    # Move to CPU for sklearn metric calculation
                    auc = roc_auc_score(y_test.cpu().numpy(), probs_test.cpu().numpy())
                    fold_scores.append(auc)
                except ImportError:
                    # Fallback to accuracy if sklearn is not available inside the loop
                    preds = (probs_test > 0.5).float()
                    acc = (preds == y_test).float().mean().item()
                    fold_scores.append(acc)

    # Return mean score across folds
    mean_score = np.mean(fold_scores)
    return torch.tensor([mean_score], dtype=torch.float32)

def c2st_auc(X: torch.Tensor, Y: torch.Tensor, **kwargs) -> torch.Tensor:
    """
    Wrapper for C2ST returning ROC AUC score.
    """
    return c2st(X, Y, scoring="roc_auc", **kwargs)