# PyTorch Toolbox: Architecture & Implementation Guide

## 1\. Project Philosophy

The goal of this repository is to decouple the **Training Logic** from the **Task Logic**.

  - **Training Logic (The Engine):** Handling loops, backpropagation, device management, logging, and saving models. This stays the same regardless of the problem.
  - **Task Logic (The Components):** Data preprocessing, Model architecture, and Loss functions. These change based on the problem (e.g., CIFAR-10 vs. Price Prediction).

## 2\. Directory Structure

Organize your GitHub repository using the following modular layout. This prevents "spaghetti code" and allows you to add new tasks (like NLP) later without breaking existing ones.

```text
pytorch-toolbox/
├── configs/                 # Configuration files (YAML/JSON) for experiments
│   ├── cifar10_config.yaml
│   └── price_pred_config.yaml
├── data/                    # Raw and processed data storage
├── src/                     # Source code
│   ├── __init__.py
│   ├── engine.py            # Contains the generic Training Loop (Trainer class)
│   ├── datasets.py          # Custom Dataset classes (Time series logic goes here)
│   ├── models.py            # Model architectures (LSTM, CNNs, ResNet)
│   └── utils.py             # Helper functions (metrics, plotting, device moving)
├── notebooks/               # For prototyping (like the file you uploaded)
├── train.py                 # Main entry point script
└── requirements.txt         # Dependencies
```

-----

## 3\. Implementation Details

### A. `src/utils.py`: Utilities & Metrics

This module holds the helper functions from your notebook, generalized for reuse.

**What to implement here:**

1.  **`moveTo(obj, device)`**: Your existing function to move tensors/lists/dicts to GPU.
2.  **Metric Factory**: A dictionary mapping string names to sklearn/torch metrics.
3.  **Plotting**: Your `plotter` function.

<!-- end list -->

```python
# src/utils.py
import torch
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score

def get_score_functions(task_type):
    if task_type == 'regression':
        return {
            'R2': r2_score,
            'MSE': mean_squared_error
        }
    elif task_type == 'classification':
        return {
            'ACC': accuracy_score
        }
    # ... add others
```

### B. `src/datasets.py`: Data Handling

This is where the logic differs between CIFAR and Price Prediction. You will create distinct classes inheriting from `torch.utils.data.Dataset`.

**1. For Price Prediction (The logic from your notebook):**
Implement a class `TimeSeriesDataset`. It should handle the windowing logic (the `sequencer` function).

```python
# src/datasets.py
import torch
from torch.utils.data import Dataset
import numpy as np

class TimeSeriesDataset(Dataset):
    def __init__(self, data, window_size, prediction_horizon=1):
        # Implement the logic from your 'sequencer' function here
        self.X, self.y = self._create_sequences(data, window_size, prediction_horizon)

    def _create_sequences(self, data, T, H):
        # Your 'sequencer' logic goes here
        pass

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.X[idx]), torch.FloatTensor(self.y[idx])
```

**2. For CIFAR-10:**
You can wrap standard `torchvision.datasets` here.

### C. `src/models.py`: Model Zoo

Store all architectures here.

```python
# src/models.py
import torch.nn as nn

# 1. Your LSTM Model (for Price Prediction)
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# 2. A CNN Model (for CIFAR-10)
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        # ... define conv layers ...
        pass
```

### D. `src/engine.py`: The Generic Training Loop

This is the core of your toolbox. It adapts your notebook's `run_epoch` function into a reusable Class. It should not care *what* the data is, only that it receives tensors.

```python
# src/engine.py
import torch
import numpy as np
from tqdm.auto import tqdm
from src.utils import moveTo

class Trainer:
    def __init__(self, model, optimizer, criterion, device, score_funcs=None, scaler=None):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.score_funcs = score_funcs if score_funcs else {}
        self.scaler = scaler # Optional: for inverse transforming regression targets

    def train_one_epoch(self, data_loader):
        self.model.train()
        running_loss = []
        y_true, y_pred = [], []

        for inputs, labels in tqdm(data_loader, desc="Training", leave=False):
            inputs = moveTo(inputs, self.device)
            labels = moveTo(labels, self.device)

            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss.append(loss.item())
            
            # Store predictions for metrics
            y_true.extend(labels.detach().cpu().numpy())
            y_pred.extend(outputs.detach().cpu().numpy())

        # Calculate metrics
        metrics = self._calculate_metrics(y_true, y_pred)
        metrics['loss'] = np.mean(running_loss)
        return metrics

    def evaluate(self, data_loader):
        self.model.eval()
        running_loss = []
        y_true, y_pred = [], []

        with torch.no_grad():
            for inputs, labels in tqdm(data_loader, desc="Validating", leave=False):
                inputs = moveTo(inputs, self.device)
                labels = moveTo(labels, self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                running_loss.append(loss.item())

                y_true.extend(labels.detach().cpu().numpy())
                y_pred.extend(outputs.detach().cpu().numpy())

        metrics = self._calculate_metrics(y_true, y_pred)
        metrics['loss'] = np.mean(running_loss)
        return metrics

    def _calculate_metrics(self, y_true, y_pred):
        # Handle inverse scaling if necessary (from your notebook logic)
        if self.scaler:
            y_true = self.scaler.inverse_transform(np.array(y_true).reshape(-1, 1))
            y_pred = self.scaler.inverse_transform(np.array(y_pred).reshape(-1, 1))
        
        results = {}
        for name, func in self.score_funcs.items():
            try:
                # Handle shape differences for Classification vs Regression
                results[name] = func(y_true, y_pred)
            except:
                pass
        return results
```

-----

## 4\. Using the Toolbox (`train.py`)

This is how you tie it all together. You can use a library like `argparse` or `hydra` to switch between modes.

**Pseudocode for `train.py`:**

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.engine import Trainer
from src.models import LSTMModel, SimpleCNN
from src.datasets import TimeSeriesDataset
from src.utils import get_score_functions

# 1. Configuration (Select Task)
TASK = "price_prediction" # or "cifar10"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Load Data & Model based on Task
if TASK == "price_prediction":
    # Load CSV, preprocess, scale (as per your notebook)
    # dataset = TimeSeriesDataset(...)
    # model = LSTMModel(input_size=4, hidden_size=100, ...)
    # criterion = nn.MSELoss()
    # score_funcs = get_score_functions('regression')
    pass

elif TASK == "cifar10":
    # dataset = torchvision.datasets.CIFAR10(...)
    # model = SimpleCNN(num_classes=10)
    # criterion = nn.CrossEntropyLoss()
    # score_funcs = get_score_functions('classification')
    pass

# 3. Setup Dataloaders
train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=100)

# 4. Initialize Engine
optimizer = torch.optim.NAdam(model.parameters(), lr=0.001)
trainer = Trainer(model, optimizer, criterion, DEVICE, score_funcs)

# 5. Run Loop
for epoch in range(150):
    train_metrics = trainer.train_one_epoch(train_loader)
    val_metrics = trainer.evaluate(val_loader)
    
    print(f"Epoch {epoch}: Train Loss {train_metrics['loss']} | Val Loss {val_metrics['loss']}")
    
    # Save best model logic here
```
5.  **Refactor Loop:** Copy the code from Section 3.D (`src/engine.py`) and verify it works with your existing LSTM logic.

Once this skeleton is working for your Price Prediction code, adding CIFAR-10 will simply require adding a new Model class and a new Dataset loader in `train.py`, without rewriting the training loop.
