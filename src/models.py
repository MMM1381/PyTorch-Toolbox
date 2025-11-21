import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Time Series Model ---
class LSTMModel(nn.Module):
    def __init__(self, input_size=4, hidden_size=256, num_layers=1, output_size=1):
        super(LSTMModel, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        out, _ = self.lstm(x) 
        
        # Take the output of the last time step
        # out shape: (batch, seq_len, hidden_size) -> slice -> (batch, hidden_size)
        out = out[:, -1, :]
        
        out = self.fc(out)
        return out

# --- Classification Model (CIFAR-10) ---
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Input: (Batch, 3, 32, 32)
        x = self.pool(F.relu(self.conv1(x))) # -> (32, 16, 16)
        x = self.pool(F.relu(self.conv2(x))) # -> (64, 8, 8)
        x = x.view(-1, 64 * 8 * 8)           # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x