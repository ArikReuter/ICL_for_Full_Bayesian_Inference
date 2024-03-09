import torch 

def MSELoss_unsqueezed(input: torch.tensor, target: torch.tensor) -> torch.tensor:
    """
    Compute MSE loss between input and target where the input is unsqueezed
    Args:
        input: torch.tensor: the input tensor
        target: torch.tensor: the target tensor
    Returns:
        torch.tensor: the MSE loss
    """
    input = input.squeeze()
    target = target.squeeze()

    assert input.shape == target.shape, f"input shape {input.shape} does not match target shape {target.shape}"

    mse  = torch.nn.MSELoss()(input, target)

    return mse