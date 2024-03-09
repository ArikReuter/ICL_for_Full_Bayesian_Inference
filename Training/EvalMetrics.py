"""
Class to implement a few evaluation metrics
"""

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import torch

def mean_squared_error_torch(y_true: torch.tensor, y_pred: torch.tensor) -> float:
    """
    Compute the mean squared error
    Args:
        y_true: torch.tensor: the true values
        y_pred: torch.tensor: the predicted values
    Returns:
        float: the mean squared error
    """
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    assert y_true.shape == y_pred.shape, f"y_true shape {y_true.shape} does not match y_pred shape {y_pred.shape}"

    return torch.tensor(mean_squared_error(y_true.detach().numpy(), y_pred.detach().numpy()), dtype=torch.float64)

def r2_score_torch(y_true: torch.tensor, y_pred: torch.tensor) -> float:
    """
    Compute the r2 score
    Args:
        y_true: torch.tensor: the true values
        y_pred: torch.tensor: the predicted values
    Returns:
        float: the r2 score
    """
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    return torch.tensor(r2_score(y_true.detach().numpy(), y_pred.detach().numpy()), dtype=torch.float64)

def mean_absolute_error_torch(y_true: torch.tensor, y_pred: torch.tensor) -> float:
    """
    Compute the mean absolute error
    Args:
        y_true: torch.tensor: the true values
        y_pred: torch.tensor: the predicted values
    Returns:
        float: the mean absolute error
    """
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    return torch.tensor(mean_absolute_error(y_true.detach().numpy(), y_pred.detach().numpy()), dtype=torch.float64)

