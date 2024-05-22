import torch 

def simulate_X_uniform(n:int, p:int, batch_size:int = 0) -> torch.tensor:
    """
    Simulate covariates from a uniform distribution on the unit interval
    Args:
        n: int: the number of observations
        p: int: the number of covariates
        batch_size: int: the batch size

    Returns:
        torch.tensor: the covariates of shape (n, p) or (batch_size, n, p)
    """

    if batch_size == 0:
        return torch.rand(n, p)
    else:
        return torch.rand(batch_size, n, p)