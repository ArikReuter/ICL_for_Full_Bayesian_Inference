"""
Class to implement a few evaluation metrics
"""

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import torch

def mean_squared_error_torch_avg(y_true: torch.tensor, y_pred: list[list[torch.tensor]]) -> float:
    """
    Compute the mean squared error averaged over the batch
    Args:
        y_true: torch.tensor: the true values
        y_pred: list[list[torch.tensor]]: the predicted values where the first list is the batch and the second list is the output that CAN contain multiple outputs
    Returns:
        float: the mean squared error
    """
    mse = 0
    for i in range(len(y_pred)):
        
        assert y_true[i].detach().numpy().shape == y_pred[i][0].detach().numpy().shape, f"y_true shape {y_true[i].detach().numpy().shape} does not match y_pred shape {y_pred[i][0].detach().numpy().shape}"
        mse += mean_squared_error(y_true[i].detach().numpy(), y_pred[i][0].detach().numpy())
    return mse / len(y_pred)
    

def r2_score_torch_avg(y_true: torch.tensor, y_pred: list[list[torch.tensor]]) -> float:
    """
    Compute the r2 score averaged over the batch
    Args:
        y_true: torch.tensor: the true values
        y_pred: list[list[torch.tensor]]: the predicted values where the first list is the batch and the second list is the output that CAN contain multiple outputs
    Returns:
        float: the r2 score
    """
    r2 = 0
    for i in range(len(y_pred)):
        assert y_true[i].detach().numpy().shape == y_pred[i][0].detach().numpy().shape, f"y_true shape {y_true[i].detach().numpy().shape} does not match y_pred shape {y_pred[i][0].detach().numpy().shape}"
        r2 += r2_score(y_true[i].detach().numpy(), y_pred[i][0].detach().numpy())
    return r2 / len(y_pred)

def mae_torch_avg(y_true: torch.tensor, y_pred: list[list[torch.tensor]]) -> float:
    """
    Compute the mean absolute error averaged over the batch
    Args:
        y_true: torch.tensor: the true values
        y_pred: list[list[torch.tensor]]: the predicted values where the first list is the batch and the second list is the output that CAN contain multiple outputs
    Returns:
        float: the mean absolute error
    """
    mae = 0
    for i in range(len(y_pred)):
        assert y_true[i].detach().numpy().shape == y_pred[i][0].detach().numpy().shape, f"y_true shape {y_true[i].detach().numpy().shape} does not match y_pred shape {y_pred[i][0].detach().numpy().shape}"
        mae += mean_absolute_error(y_true[i].detach().numpy(), y_pred[i][0].detach().numpy())
    return mae / len(y_pred)
