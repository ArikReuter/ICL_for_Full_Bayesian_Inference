import pyro 
from pyro.infer import MCMC, NUTS, HMC
from pyro.infer.mcmc.util import predictive
from pyro.infer import Predictive
import torch

from PFNExperiments.LinearRegression.GenerativeModels.LM_abstract import ppgram_linear_model_return_y
from PFNExperiments.LinearRegression.ComparisonModels.PosteriorComparisonModel import PosteriorComparisonModel

class Hamiltionian_MC(PosteriorComparisonModel):
    """
    use Hamiltonian Monte Carlo to obtain posterior samples for a given probabilistic program
    """

    def __init__(self, 
                 pprogram: ppgram_linear_model_return_y,
                 n_samples:int = 200,
                 n_warmup:int = 100,
                 kernel_kwargs: dict = {},
                 mcmc_kwargs: dict = {},
                 shuffle_samples: bool = True

                 ) -> None:
        
        """
        Args:
            pprogram: ppgram_linear_model_return_y: the probabilistic program
            n_samples: int: the number of samples to draw from the posterior
            n_warmup: int: the number of warmup samples to draw from the posterior
            kernel_kwargs: dict: the keyword arguments for the kernel
            mcmc_kwargs: dict: the keyword arguments for the mcmc
            shuffle_samples: bool: whether to shuffle the samples
        
        Returns:
            None
        """
        self.pprogram = pprogram
        self.n_samples = n_samples
        self.n_warmup = n_warmup

        if "adapt_step_size" not in kernel_kwargs:
            kernel_kwargs["adapt_step_size"] = True

        self.nuts_kernel = NUTS(self.pprogram, **kernel_kwargs)
        self.mcmc = MCMC(self.nuts_kernel, num_samples=self.n_samples, warmup_steps=self.n_warmup, **mcmc_kwargs, disable_progbar = True)
        self.shuffle_samples = shuffle_samples


    def sample_posterior(self,  
                X: torch.Tensor,
                y: torch.Tensor) -> torch.Tensor:
        """
        A method that samples from the posterior distribution
        Args:
            X: torch.Tensor: the covariates
            y: torch.Tensor: the response variable
        Returns:
            torch.Tensor: the samples from the posterior distribution
        """
        self.mcmc.run(X, y)
        posterior_samples = self.mcmc.get_samples()

        if self.shuffle_samples:
            posterior_samples = {k: v[torch.randperm(v.shape[0])] for k, v in posterior_samples.items()}

        return posterior_samples
    

    def __repr__(self) -> str:
        return "Hamiltonian Monte Carlo"