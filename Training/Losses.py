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


def nll_loss_full_gaussian(pred, target):
  """
  NLL based on a full covariance matrix
  Args: 
    pred: tuple: the prediction tuple, contains the mean, covariance factor and diagonal for covariance matrix
    target: torch.tensor: the target tensor
  """

  mu, cov_factor, cov_diag = pred
  cov_factor = cov_factor.reshape(mu.shape[0], mu.shape[1], -1)
  cov_diag = cov_diag **2 + 1e-5
  dist = torch.distributions.LowRankMultivariateNormal(
      loc = mu,
      cov_factor = cov_factor,
      cov_diag = cov_diag
      )

  nll = - dist.log_prob(target)
  nll = torch.mean(nll)

  return nll