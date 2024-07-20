from torch import nn
import torch

class RBF(nn.Module):

    def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth=None):
        super().__init__()
        self.bandwidth_multipliers = mul_factor ** (torch.arange(n_kernels) - n_kernels // 2)
        self.bandwidth = bandwidth

    def get_bandwidth(self, L2_distances):
        if self.bandwidth is None:
            n_samples = L2_distances.shape[0]
            return L2_distances.data.sum() / (n_samples ** 2 - n_samples)

        return self.bandwidth

    def forward(self, X):
        L2_distances = torch.cdist(X, X) ** 2
        return torch.exp(-L2_distances[None, ...] / (self.get_bandwidth(L2_distances) * self.bandwidth_multipliers)[:, None, None]).sum(dim=0)


class MMDLoss(nn.Module):

    def __init__(self, kernel=RBF()):
        super().__init__()
        self.kernel = kernel

    def forward(self, X, Y):
        K = self.kernel(torch.vstack([X, Y]))

        X_size = X.shape[0]
        XX = K[:X_size, :X_size].mean()
        XY = K[:X_size, X_size:].mean()
        YY = K[X_size:, X_size:].mean()
        return XX - 2 * XY + YY
    


def compare_samples_mmd(P: torch.tensor, Q: torch.tensor, n_kernels = 5, mul_factor = 2.0, bandwith = None) -> dict:
    """
    Compare two samples using the MMD loss
    Args:
        P: torch.tensor: the samples from the first distribution
        Q: torch.tensor: the samples from the second distribution
        n_kernels: int: the number of kernels to use
        mul_factor: float: the factor to multiply the bandwidths with
        bandwith: float: the bandwidth to use
    Returns:
        dict: a dictionary containing the result of the comparison
    """
    try:
        mmd = MMDLoss(kernel=RBF(n_kernels=n_kernels, mul_factor=mul_factor, bandwidth=bandwith))
        mmd_value = mmd(P, Q).item()
        res = {"MMD": mmd_value}
    except Exception as e:
        print(f"An exception occured in compare_samples_mmd: {e}")
        res = {"MMD": torch.nan}
    
    return res