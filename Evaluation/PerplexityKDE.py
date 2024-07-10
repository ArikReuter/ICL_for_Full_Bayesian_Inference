import torch 
from scipy.stats import gaussian_kde
import numpy as np

class PerplexityKDE():
    """
    Evaluate the perplexity of the model based on the likelihood of a KDE estimated with samples and evaluated at the true parameters
    """

    def __init__(
            self,
            bw_method = None,
    ):
        """
        Args:
            bw_method: str: the method to use to estimate the bandwidth of the KDE
        """

        self.bw_method = bw_method
        
    def __call__(
            self,
            samples: torch.tensor,
            true_parameter: torch.tensor,
    ) -> float:
        """
        Evluate the perplexity of the model based on the likelihood of a KDE estimated with samples and evaluated at the true parameters
        Args:
            samples: torch.tensor: the samples to use of shape (n_samples, n_dims)
            true_parameter: torch.tensor: the true parameters to evaluate the likelihood at of shape (n_dims)

        Returns:
            float: the perplexity of the model measured as the log likelihood of the true parameter
        """


        true_parameter = true_parameter.squeeze().numpy()
        samples = samples.numpy()

        assert len(true_parameter) == samples.shape[1], "The number of dimensions of the true parameter and the samples should match. but got {} and {}".format(len(true_parameter), samples.shape[1])

        kde = gaussian_kde(samples.T, bw_method=self.bw_method)

        log_pdf = kde.logpdf(true_parameter)

        return log_pdf
    

def compare_to_gt_perplexity_kde(
        samples: torch.tensor,
        true_parameter: torch.tensor,
        bw_method = None,
) -> float:
    """
    Compare the perplexity of the model based on the likelihood of a KDE estimated with samples and evaluated at the true parameters
    Args:
        samples: torch.tensor: the samples to use of shape (n_samples, n_dims)
        true_parameter: torch.tensor: the true parameters to evaluate the likelihood at of shape (n_dims)
        bw_method: str: the method to use to estimate the bandwidth of the KDE

    Returns:
        float: the perplexity of the model measured as the log likelihood of the true parameter
    """

    metric = PerplexityKDE(bw_method=bw_method)

    return metric(samples, true_parameter)


