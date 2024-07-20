import torch 
from scipy.stats import gaussian_kde
from scipy.optimize import minimize


def compare_to_gt_mean_difference(samples:torch.tensor, true_parameter:torch.tensor) -> float:
    """
    compute how close the mean of the samples is to the true parameter
    Args:
        samples: torch.tensor: the samples to use of shape (n_samples, n_dims)
        true_parameter: torch.tensor: the true parameters to evaluate the likelihood at of shape (n_dims)
    Returns:
        float: the mean difference between the samples and the true parameter
    """
    samples = samples.detach().cpu()

    res = torch.mean(torch.abs(samples.mean(dim = 0) - true_parameter)).item()
    return res.item()

def compare_to_gt_MAP(samples:torch.tensor, true_parameter:torch.tensor, kernel_bandwidth = None) -> float:
    """
    compute how close the MAP of the samples is to the true parameter. The MAP is computed using a KDE
    Args:
        samples: torch.tensor: the samples to use of shape (n_samples, n_dims)
        true_parameter: torch.tensor: the true parameters to evaluate the likelihood at of shape (n_dims)
        kernel_bandwidth: float: the bandwidth to use for the KDE
    Returns:
        float: the MAP difference between the samples and the true parameter
    """
    
    # compute the MAP of the samples

    try: 
        samples = samples.detach().cpu().numpy()
    
    except:
        pass

    kde = gaussian_kde(samples.T, bw_method=kernel_bandwidth)

    # get the MAP

    def objective_function(x):
        return -kde.logpdf(x)

    MAP = minimize(objective_function, samples.mean(axis = 0)).x
    MAP = torch.tensor(MAP)

    res = torch.mean(torch.abs(MAP - true_parameter))

    return res.item()




