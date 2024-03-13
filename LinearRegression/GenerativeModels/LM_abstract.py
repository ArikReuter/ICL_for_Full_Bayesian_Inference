from abc import ABC, abstractmethod
import torch 
import inspect

@abstractmethod
def pprogram_linear_model_return_dict(x: torch.tensor, y: torch.tensor = None) -> dict:
    """
    A probabilistic program that models a linear model. x denotes the covariates and y denotes the response variable.
    Args: 
        x: torch.tensor: the covariates
        y: torch.tensor: the response variable
    Returns:
        dict: a dictionary containing the parameters of the linear model. Contains at least the following keys:
        {
            "x": torch.tensor: the covariates
            "y": torch.tensor: the response variable
            "beta": torch.tensor: the parameters of the linear model
        }
    """
    pass 

@abstractmethod
def ppgram_linear_model_return_y(x: torch.tensor,  y: torch.tensor = None) -> torch.tensor:
    """
    A probabilistic program that models a linear model. x denotes the covariates and y denotes the response variable.
    Args: 
        x: torch.tensor: the covariates
    Returns:
        torch.tensor: the response variable
    """
    pass

def return_only_y(pprogram: pprogram_linear_model_return_dict) -> ppgram_linear_model_return_y:
    """
    A decorator that ensures a ppogram returns only the response variable y.
    Args:
        pprogram: pprogram_linear_model_return_dict: the probabilistic program
    Returns:
        callable: a callable that returns only the response variable y
    """
    def wrapper(x: torch.tensor, y: torch.tensor = None) -> torch.tensor:
        return pprogram(x, y)["y"]
    
    return wrapper

def pprogram_X(n: int, p:int) -> torch.Tensor:
    """
    A probabilistic program that simulates covariates. 
    Args:
        n: int: the number of observations
        p: int: the number of covariates
    Returns:
        torch.tensor: the covariates of shape (n, p)
    """
    pass

def print_code(fun: callable) -> None:
    """
    Print the source code of a function
    Args:
        fun: callable: the function
    """
    print(inspect.getsource(fun))