import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error, accuracy_score

def moveTo(obj, device):
    """
    Recursively moves tensors, lists, tuples, or dictionaries to the specified device.
    """
    if isinstance(obj, list):
        return [moveTo(x, device) for x in obj]
    elif isinstance(obj, tuple):
        return tuple(moveTo(x, device) for x in obj)
    elif isinstance(obj, dict):
        return {k: moveTo(v, device) for k, v in obj.items()}
    elif hasattr(obj, "to"):
        return obj.to(device)
    else:
        return obj

def get_score_functions(task_type):
    """
    Returns a dictionary of metric functions based on the task type.
    """
    if task_type == 'regression':
        return {
            'MAE': mean_absolute_error,
            'MSE': mean_squared_error,
            'RMSE': root_mean_squared_error,
            'R2': r2_score,
        }
    elif task_type == 'classification':
        return {
            'Accuracy': accuracy_score
        }
    else:
        raise ValueError(f"Unknown task type: {task_type}")

def plotter(history, threshold=None, save_path=None):
    """
    Plots training and validation metrics from the history dictionary.
    """
    metrics = [k.split()[1] for k in history.keys() if 'train' in k and 'loss' not in k]
    
    # Plot Loss
    plt.figure(figsize=(10, 5))
    plt.plot(history['train loss'], label='Train Loss')
    plt.plot(history['val loss'], label='Val Loss')
    plt.title('Loss over Epochs')
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(f"{save_path}_loss.png")
    plt.show()

    # Plot other metrics
    for m in metrics:
        plt.figure(figsize=(10, 5))
        train_key = f'train {m}'
        val_key = f'val {m}'
        
        if train_key in history:
            t_data = np.array(history[train_key])
            # Apply threshold clipping if provided (visual cleanup)
            if threshold is not None:
                t_data[t_data > threshold] = threshold
                t_data[t_data < -threshold] = -threshold
            plt.plot(t_data, label=train_key)
            
        if val_key in history:
            v_data = np.array(history[val_key])
            if threshold is not None:
                v_data[v_data > threshold] = threshold
                v_data[v_data < -threshold] = -threshold
            plt.plot(v_data, label=val_key)
            
        plt.title(f'{m} over Epochs')
        plt.legend()
        plt.grid(True)
        if save_path:
            plt.savefig(f"{save_path}_{m}.png")
        plt.show()