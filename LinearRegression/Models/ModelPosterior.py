from abc import ABC, abstractmethod
import torch
from torch.distributions import LowRankMultivariateNormal
from numpy.linalg import LinAlgError

from PFNExperiments.Training.EvalMetrics import mean, median, std


class ModelPosterior(ABC):
    """
    Class that takes in a model and returns the posterior the model provides
    """

    def pred2posterior(self, pred) -> torch.distributions.distribution.Distribution:
        """
        A method that takes in the models prediction and returns the posterior distribution
        Args:
            pred: the models predicctions 
        Returns:
            torch.distributions.distribution.Distribution: the posterior distribution
        """
        pass

    @abstractmethod
    def negative_log_likelihood(self, pred, target: torch.Tensor) -> torch.Tensor:
        """
        A method that takes in the models prediction and the target and returns the negative log likelihood
        Args:
            pred: torch.Tensor: the models prediction
            target: torch.Tensor: the target
        Returns:
            torch.Tensor: the negative log likelihood
        """
        pass

    def negative_log_likelihood_list(self, list_of_targets: list[torch.Tensor], list_of_preds) -> list[torch.Tensor]:
        """
        A method that takes in the models prediction and the target and returns the negative log likelihood averaged over the batch
        Args:
            list_of_preds: list[torch.Tensor]: the models prediction
            list_of_targets: list[torch.Tensor]: the target
        Returns:
            torch.Tensor: the negative log likelihood
        """
        nll = []
        assert len(list_of_preds) == len(list_of_targets), "The length of list_of_preds does not match the length of list_of_targets"
        for pred, target in zip(list_of_preds, list_of_targets):
            nll.append(self.negative_log_likelihood(pred, target))

        return nll
    
    def negative_log_likelihood_avg(self, list_of_targets: list[torch.Tensor], list_of_preds) -> torch.Tensor:
        """
        A method that takes in the models prediction and the target and returns the negative log likelihood averaged over the batch
        Args:
            list_of_preds: list[torch.Tensor]: the models prediction
            list_of_targets: list[torch.Tensor]: the target
        
        Returns:
            torch.Tensor: the negative log likelihood
        """
        nll = self.negative_log_likelihood_list(list_of_targets, list_of_preds)
        return mean(nll)
    
    def negative_log_likelihood_std(self, list_of_targets: list[torch.Tensor], list_of_preds) -> torch.Tensor:
        """
        A method that takes in the models prediction and the target and returns the standard deviation of the negative log likelihood over the batch
        Args:
            list_of_preds: list[torch.Tensor]: the models prediction
            list_of_targets: list[torch.Tensor]: the target
        Returns:
            torch.Tensor: the negative log likelihood
        """
        nll = self.negative_log_likelihood_list(list_of_targets, list_of_preds)
        return std(nll)
    
    def negative_log_likelihood_median(self, list_of_targets: list[torch.Tensor], list_of_preds) -> torch.Tensor:
        """
        A method that takes in the models prediction and the target and returns the median of the negative log likelihood over the batch
        Args:
            list_of_preds: list[torch.Tensor]: the models prediction
            list_of_targets: list[torch.Tensor]: the target
        Returns:
            torch.Tensor: the negative log likelihood
        """
        nll = self.negative_log_likelihood_list(list_of_targets, list_of_preds)
        return median(nll)
    

    def pred2posterior_samples(self, pred, n_samples: int) -> torch.Tensor:
        """
        A method that takes in the models prediction and returns samples from the posterior distribution
        Args:
            pred: torch.Tensor: the models prediction
            n_samples: int: the number of samples
        Returns:
            torch.Tensor: the samples from the posterior distribution with shape (n_samples, ...) where ... denotes the shape of the samples
        """
        pass




class ModelPosteriorFullGaussian(ModelPosterior):
    """
    A class that takes in a model and returns the posterior the model provides
    """

    def __init__(self, cov_reg_factor: float = 0.0, loss_on_error: float = 1e10):
        """
        Args:
            cov_reg_factor: float: the regularization factor for the covariance matrix
        """
        self.cov_reg_factor = cov_reg_factor
        self.loss_on_error = loss_on_error


    
    def pred2posterior(self, pred) -> torch.distributions.distribution.Distribution:
        """
        A method that takes in the models prediction and returns the posterior distribution
        Args:
            pred: the models predicctions 
        Returns:
            torch.distributions.distribution.Distribution: the posterior distribution
        """


        mu, cov_factor, cov_diag = pred
        cov_factor = cov_factor.reshape(mu.shape[0], mu.shape[1], -1)
        cov_diag = cov_diag **2 + self.cov_reg_factor
        dist = torch.distributions.LowRankMultivariateNormal(
            loc = mu,
            cov_factor = cov_factor,
            cov_diag = cov_diag
            )
        return dist

    def negative_log_likelihood(self, pred, target: torch.Tensor) -> torch.Tensor:
        """
        A method that takes in the models prediction and the target and returns the negative log likelihood
        Args:
            pred: torch.Tensor: the models prediction
            target: torch.Tensor: the target
        Returns:
            torch.Tensor: the negative log likelihood
        """

        if type(target) == list:
            assert len(target) == 1, "The target is a list but the length is not 1"
            target = target[0]
        
        try: 
            dist = self.pred2posterior(pred)

            nll = - dist.log_prob(target)

        except RuntimeError as e:
            if 'linalg.cholesky' in str(e):
                print("Caught a _LinAlgError related to Cholesky decomposition: The input is not positive-definite.")
            else:
                raise

            mu, cov_factor, cov_diag = pred
            return torch.tensor(self.loss_on_error, requires_grad = True) + 1/torch.sum(cov_diag **2) + 1/torch.sum(cov_factor **2) + 1/torch.sum(mu **2)

        return nll.mean()
    
    def pred2posterior_samples(self, pred, n_samples: int) -> torch.Tensor:
        """
        A method that takes in the models prediction and returns samples from the posterior distribution
        Args:
            pred: torch.Tensor: the models prediction
            n_samples: int: the number of samples
        Returns:
            torch.Tensor: the samples from the posterior distribution with shape (n_samples, ...) where ... denotes the shape of the samples
        """
        dist = self.pred2posterior(pred)
        samples = dist.sample((n_samples,))
        return samples