import torch
from torch.utils.data import Dataset
import numpy as np
from torchvision import datasets, transforms

class TimeSeriesDataset(Dataset):
    def __init__(self, data, window_size, overlap=1, target_col=-1):
        """
        Args:
            data (np.array): The scaled input data matrix (N_samples, N_features).
            window_size (int): The lookback period (T).
            overlap (int): Step size for sliding window.
            target_col (int): Index of the column to predict (usually the last one).
        """
        self.window_size = window_size
        self.overlap = overlap
        self.target_col = target_col
        
        # Convert to float32 immediately to avoid double precision issues in PyTorch
        self.data = data.astype(np.float32)
        
        self.X, self.y = self._sequencer()

    def _sequencer(self):
        X = []
        y = []
        length = len(self.data)
        
        for i in range(0, length, self.overlap):
            # Check if we have enough data for the window + target
            if length - 1 < i + self.window_size:
                break
                
            # Create sequence
            x_vector = self.data[i : i + self.window_size]
            
            # Target is the value at the next step after the window
            # Note: Your notebook used `data[i+T][-1]`
            target = self.data[i + self.window_size][self.target_col]
            
            X.append(x_vector)
            y.append(target)
            
        return np.array(X), np.array(y).reshape(-1, 1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # Returns tensors suitable for the model
        return torch.from_numpy(self.X[idx]), torch.from_numpy(self.y[idx])

def get_cifar10_datasets(root='./data'):
    """
    Returns train and test datasets for CIFAR-10 with standard transforms.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = datasets.CIFAR10(root=root, train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root=root, train=False, download=True, transform=transform)
    
    return train_dataset, test_dataset