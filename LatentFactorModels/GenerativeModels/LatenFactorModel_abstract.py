import torch 

def return_only_x(pprogram):
    """
    A decorator that ensures a ppogram returns only the data x
    Args:
        pprogram: pprogram_linear_model_return_dict: the probabilistic program
    Returns:
        callable: a callable that returns only the response variable y
    """
    def wrapper(x: torch.tensor) -> torch.tensor:
        return pprogram(x)["x"]
    
    return wrapper