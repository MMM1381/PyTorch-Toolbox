import torch
import numpy as np
import time
from tqdm.auto import tqdm
from src.utils import moveTo

class Trainer:
    def __init__(self, model, optimizer, criterion, device, score_funcs=None, scaler=None):
        """
        Args:
            model: PyTorch model.
            optimizer: PyTorch optimizer.
            criterion: Loss function.
            device: 'cuda' or 'cpu'.
            score_funcs (dict): Dictionary of metric functions (e.g., {'R2': r2_score}).
            scaler: Sklearn scaler object (optional) for inverse transforming regression targets.
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.score_funcs = score_funcs if score_funcs else {}
        self.scaler = scaler
        
        # History container
        self.history = {
            "train loss": [], "val loss": [], "epoch": [], "time": []
        }
        for name in self.score_funcs.keys():
            self.history[f"train {name}"] = []
            self.history[f"val {name}"] = []

    def _run_epoch(self, data_loader, is_training=True, desc=None):
        self.model.train() if is_training else self.model.eval()
        
        running_loss = []
        y_true = []
        y_pred = []
        
        start_time = time.time()
        
        # Disable gradient calculation for validation
        with torch.set_grad_enabled(is_training):
            for inputs, labels in tqdm(data_loader, desc=desc, leave=False):
                inputs = moveTo(inputs, self.device)
                labels = moveTo(labels, self.device)

                # Forward Pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                if is_training:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                running_loss.append(loss.item())

                # Collect data for metrics
                # Detach and move to CPU
                labels_np = labels.detach().cpu().numpy()
                outputs_np = outputs.detach().cpu().numpy()

                # If we have a scaler (Regression), inverse transform
                if self.scaler:
                    # Handle potential shape mismatch depending on scaler setup
                    # Assuming scaler was fit on the target column shape (-1, 1)
                    if outputs_np.ndim == 1: outputs_np = outputs_np.reshape(-1, 1)
                    if labels_np.ndim == 1: labels_np = labels_np.reshape(-1, 1)
                    
                    outputs_np = self.scaler.inverse_transform(outputs_np)
                    labels_np = self.scaler.inverse_transform(labels_np)
                
                # For classification, get class indices if output is raw logits
                # (Assuming labels are not one-hot for metrics like accuracy)
                if not self.scaler and outputs_np.shape[1] > 1: 
                     outputs_np = np.argmax(outputs_np, axis=1)

                y_true.extend(labels_np.tolist())
                y_pred.extend(outputs_np.tolist())

        end_time = time.time()
        epoch_loss = np.mean(running_loss)
        
        # Calculate custom metrics
        metrics = {}
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        for name, func in self.score_funcs.items():
            try:
                # Flatten arrays if necessary for sklearn metrics
                if self.scaler:
                    metrics[name] = func(y_true.flatten(), y_pred.flatten())
                else:
                    metrics[name] = func(y_true, y_pred)
            except Exception as e:
                print(f"Error calculating {name}: {e}")
                metrics[name] = float('nan')

        return epoch_loss, metrics, end_time - start_time

    def fit(self, train_loader, val_loader, num_epochs, save_path="best_model.pt"):
        best_val_loss = float('inf')
        
        print(f"Starting training on {self.device}...")
        
        for epoch in range(num_epochs):
            # Training
            train_loss, train_metrics, train_time = self._run_epoch(
                train_loader, is_training=True, desc=f"Epoch {epoch+1} Train"
            )
            
            # Validation
            val_loss, val_metrics, _ = self._run_epoch(
                val_loader, is_training=False, desc=f"Epoch {epoch+1} Val"
            )
            
            # Update History
            self.history["epoch"].append(epoch + 1)
            self.history["time"].append(train_time)
            self.history["train loss"].append(train_loss)
            self.history["val loss"].append(val_loss)
            
            for k, v in train_metrics.items(): self.history[f"train {k}"].append(v)
            for k, v in val_metrics.items(): self.history[f"val {k}"].append(v)

            # --- Custom Print Format ---
            print(f'Epoch {epoch+1}/{num_epochs}:')
            
            # Train Line
            train_metrics_str = ', '.join([
                f'{k}: {v[-1]:.4f}' for k, v in self.history.items() 
                if k.startswith("train ") and k != "train loss"
            ])
            # We use regex or explicit check to avoid double comma if no metrics exist
            train_connector = ", " if train_metrics_str else ""
            print(f'Train - Loss: {self.history["train loss"][-1]:.4f}{train_connector}{train_metrics_str}')

            # Val Line
            val_metrics_str = ', '.join([
                f'{k}: {v[-1]:.4f}' for k, v in self.history.items() 
                if k.startswith("val ") and k != "val loss"
            ])
            val_connector = ", " if val_metrics_str else ""
            print(f'Val   - Loss: {self.history["val loss"][-1]:.4f}{val_connector}{val_metrics_str}')

            # Save Best Model (based on Val Loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), save_path)
                
        print("Training Complete.")
        return self.history