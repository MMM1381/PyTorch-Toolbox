import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.engine import Trainer
from src.models import LSTMModel, SimpleCNN
from src.datasets import TimeSeriesDataset, get_cifar10_datasets
from src.utils import get_score_functions, plotter

# --- Configuration ---
TASK = "price"  # Options: "price" or "cifar10"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 100
NUM_EPOCHS = 5 # Reduced for demo purposes, set to 150 for real run

def run_price_prediction():
    print("--- Running Price Prediction Task ---")
    
    # 1. Dummy Data Generation (Replace this with pd.read_csv loading)
    # Simulating your DataFrame with Open, High, Low, Close
    dates = pd.date_range(start='1/1/2020', periods=1000, freq='T')
    df = pd.DataFrame(np.random.rand(1000, 4), columns=['Open', 'High', 'Low', 'Close'], index=dates)
    
    # 2. Preprocessing
    # Scale Features
    feature_scaler = MinMaxScaler()
    data_scaled = feature_scaler.fit_transform(df.values)
    
    # Scale Target (for inverse transform later)
    # Assuming 'Close' is the last column (index 3)
    target_scaler = MinMaxScaler()
    target_scaler.fit(df['Close'].values.reshape(-1, 1))

    # 3. Datasets
    # Split Data
    train_size = int(len(data_scaled) * 0.8)
    train_data = data_scaled[:train_size]
    val_data = data_scaled[train_size:]
    
    # Create Dataset objects
    T = 50 # Window size
    train_dataset = TimeSeriesDataset(train_data, window_size=T, target_col=3)
    val_dataset = TimeSeriesDataset(val_data, window_size=T, target_col=3)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 4. Model
    model = LSTMModel(input_size=4, hidden_size=100, num_layers=1, output_size=1)
    
    # 5. Setup Trainer
    criterion = nn.MSELoss()
    optimizer = torch.optim.NAdam(model.parameters(), lr=0.001)
    score_funcs = get_score_functions('regression')
    
    trainer = Trainer(
        model=model, 
        optimizer=optimizer, 
        criterion=criterion, 
        device=DEVICE, 
        score_funcs=score_funcs,
        scaler=target_scaler # Pass scaler for inverse transform in metrics
    )
    
    # 6. Train
    history = trainer.fit(train_loader, val_loader, num_epochs=NUM_EPOCHS, save_path="price_model.pt")
    
    # 7. Visualize
    plotter(history)

def run_cifar10():
    print("--- Running CIFAR-10 Classification Task ---")
    
    # 1. Datasets
    train_dataset, val_dataset = get_cifar10_datasets()
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 2. Model
    model = SimpleCNN(num_classes=10)
    
    # 3. Setup Trainer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    score_funcs = get_score_functions('classification')
    
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=DEVICE,
        score_funcs=score_funcs,
        scaler=None # No scaler needed for classification labels
    )
    
    # 4. Train
    history = trainer.fit(train_loader, val_loader, num_epochs=NUM_EPOCHS, save_path="cifar_model.pt")
    
    # 5. Visualize
    plotter(history)

if __name__ == "__main__":
    if TASK == "price":
        run_price_prediction()
    elif TASK == "cifar10":
        run_cifar10()