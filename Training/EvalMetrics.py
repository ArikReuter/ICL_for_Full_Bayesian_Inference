"""
Class to implement a few evaluation metrics
"""

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import torch
import numpy as np

def mean(list: list[float]) -> float:
    """
    Compute the mean of a list
    Args:
        list: list[float]: the list of numbers
    Returns:
        float: the mean
    """
    return sum(list) / len(list)

def std(list: list[float]) -> float:
    """
    Compute the standard deviation of a list
    Args:
        list: list[float]: the list of numbers
    Returns:
        float: the standard deviation
    """
    m = mean(list)
    return (sum((x - m) ** 2 for x in list) / len(list)) ** 0.5

def median(list: list[float]) -> float:
    """
    Compute the median of a list
    Args:
        list: list[float]: the list of numbers
    Returns:
        float: the median
    """
    median = np.median(list)

    return median


def mean_squared_error_torch_list(y_true: torch.tensor, y_pred: list[list[torch.tensor]]) -> list[float]:
    """
    Compute the mean squared error for each output
    Args:
        y_true: torch.tensor: the true values
        y_pred: list[list[torch.tensor]]: the predicted values where the first list is the batch and the second list is the output that CAN contain multiple outputs
    Returns:
        list[float]: the mean squared error for each output
    """
    mse = []
    for i in range(len(y_pred[0])):
        mse.append(mean_squared_error(y_true.detach().numpy(), y_pred[0][i].detach().numpy()))
    return mse

def mean_squared_error_torch_avg(y_true: torch.tensor, y_pred: list[list[torch.tensor]]) -> float:
    """
    Compute the mean squared error averaged over all outputs
    Args:
        y_true: torch.tensor: the true values
        y_pred: list[list[torch.tensor]]: the predicted values where the first list is the batch and the second list is the output that CAN contain multiple outputs
    Returns:
        float: the mean squared error averaged over all outputs
    """
    mse_list = mean_squared_error_torch_list(y_true, y_pred)
    mean_mse = mean(mse_list)
    return mean_mse

def mean_squared_error_torch_std(y_true: torch.tensor, y_pred: list[list[torch.tensor]]) -> float:
    """
    Compute the standard deviation of the mean squared error over all outputs
    Args:
        y_true: torch.tensor: the true values
        y_pred: list[list[torch.tensor]]: the predicted values where the first list is the batch and the second list is the output that CAN contain multiple outputs
    Returns:
        float: the standard deviation of the mean squared error over all outputs
    """
    mse_list = mean_squared_error_torch_list(y_true, y_pred)
    std_mse = std(mse_list)
    return std_mse

def mean_squared_error_torch_median(y_true: torch.tensor, y_pred: list[list[torch.tensor]]) -> float:
    """
    Compute the median of the mean squared error over all outputs
    Args:
        y_true: torch.tensor: the true values
        y_pred: list[list[torch.tensor]]: the predicted values where the first list is the batch and the second list is the output that CAN contain multiple outputs
    Returns:
        float: the median of the mean squared error over all outputs
    """
    mse_list = mean_squared_error_torch_list(y_true, y_pred)
    median_mse = median(mse_list)
    return median_mse

def r2_score_torch_list(y_true: torch.tensor, y_pred: list[list[torch.tensor]]) -> list[float]:
    """
    Compute the r2 score for each output
    Args:
        y_true: torch.tensor: the true values
        y_pred: list[list[torch.tensor]]: the predicted values where the first list is the batch and the second list is the output that CAN contain multiple outputs
    Returns:
        list[float]: the r2 score for each output
    """
    r2 = []
    for i in range(len(y_pred[0])):
        r2.append(r2_score(y_true.detach().numpy(), y_pred[0][i].detach().numpy()))
    return r2

def r2_score_torch_avg(y_true: torch.tensor, y_pred: list[list[torch.tensor]]) -> float:
    """
    Compute the r2 score averaged over all outputs
    Args:
        y_true: torch.tensor: the true values
        y_pred: list[list[torch.tensor]]: the predicted values where the first list is the batch and the second list is the output that CAN contain multiple outputs
    Returns:
        float: the r2 score averaged over all outputs
    """
    r2_list = r2_score_torch_list(y_true, y_pred)
    mean_r2 = mean(r2_list)
    return mean_r2

def r2_score_torch_std(y_true: torch.tensor, y_pred: list[list[torch.tensor]]) -> float:
    """
    Compute the standard deviation of the r2 score over all outputs
    Args:
        y_true: torch.tensor: the true values
        y_pred: list[list[torch.tensor]]: the predicted values where the first list is the batch and the second list is the output that CAN contain multiple outputs
    Returns:
        float: the standard deviation of the r2 score over all outputs
    """
    r2_list = r2_score_torch_list(y_true, y_pred)
    std_r2 = std(r2_list)
    return std_r2

def r2_score_torch_median(y_true: torch.tensor, y_pred: list[list[torch.tensor]]) -> float:
    """
    Compute the median of the r2 score over all outputs
    Args:
        y_true: torch.tensor: the true values
        y_pred: list[list[torch.tensor]]: the predicted values where the first list is the batch and the second list is the output that CAN contain multiple outputs
    Returns:
        float: the median of the r2 score over all outputs
    """
    r2_list = r2_score_torch_list(y_true, y_pred)
    median_r2 = median(r2_list)
    return median_r2

def mae_torch_list(y_true: torch.tensor, y_pred: list[list[torch.tensor]]) -> list[float]:
    """
    Compute the mean absolute error for each output
    Args:
        y_true: torch.tensor: the true values
        y_pred: list[list[torch.tensor]]: the predicted values where the first list is the batch and the second list is the output that CAN contain multiple outputs
    Returns:
        list[float]: the mean absolute error for each output
    """
    mae = []
    for i in range(len(y_pred[0])):
        mae.append(mean_absolute_error(y_true.detach().numpy(), y_pred[0][i].detach().numpy()))
    return mae


def mae_torch_avg(y_true: torch.tensor, y_pred: list[list[torch.tensor]]) -> float:
    """
    Compute the mean absolute error averaged over all outputs
    Args:
        y_true: torch.tensor: the true values
        y_pred: list[list[torch.tensor]]: the predicted values where the first list is the batch and the second list is the output that CAN contain multiple outputs
    Returns:
        float: the mean absolute error averaged over all outputs
    """
    mae_list = mae_torch_list(y_true, y_pred)
    mean_mae = mean(mae_list)
    return mean_mae

def mae_torch_std(y_true: torch.tensor, y_pred: list[list[torch.tensor]]) -> float:
    """
    Compute the standard deviation of the mean absolute error over all outputs
    Args:
        y_true: torch.tensor: the true values
        y_pred: list[list[torch.tensor]]: the predicted values where the first list is the batch and the second list is the output that CAN contain multiple outputs
    Returns:
        float: the standard deviation of the mean absolute error over all outputs
    """
    mae_list = mae_torch_list(y_true, y_pred)
    std_mae = std(mae_list)
    return std_mae

def mae_torch_median(y_true: torch.tensor, y_pred: list[list[torch.tensor]]) -> float:
    """
    Compute the median of the mean absolute error over all outputs
    Args:
        y_true: torch.tensor: the true values
        y_pred: list[list[torch.tensor]]: the predicted values where the first list is the batch and the second list is the output that CAN contain multiple outputs
    Returns:
        float: the median of the mean absolute error over all outputs
    """
    mae_list = mae_torch_list(y_true, y_pred)
    median_mae = median(mae_list)
    return median_mae