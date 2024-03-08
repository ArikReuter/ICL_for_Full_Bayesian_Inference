import torch 

def simulate_X_uniform(n:int, p:int) -> torch.tensor:
    """
    Simulate covariates from a uniform distribution on the unit interval
    Args:
        n: int: the number of observations
        p: int: the number of covariates

    Returns:
        torch.tensor: the covariates of shape (n, p)
    """

    return torch.rand(n, p)