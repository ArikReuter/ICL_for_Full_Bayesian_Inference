import pyro 
from pyro.infer import MCMC, NUTS, HMC
from pyro.infer.mcmc.util import predictive
from pyro.infer import Predictive
import torch

from PFNExperiments.LinearRegression.GenerativeModels.LM_abstract import ppgram_linear_model_return_y

class Hamiltionian_MC:
    """
    use Hamiltonian Monte Carlo to obtain posterior samples for a given probabilistic program
    """

    def __init__(self, 
                 pprogram: ppgram_linear_model_return_y,
                 n_samples:int = 200,
                 n_warmup:int = 100) -> None:
        
        """
        Args:
            pprogram: ppgram_linear_model_return_y: the probabilistic program
            n_samples: int: the number of samples to draw from the posterior
            n_warmup: int: the number of warmup samples to draw from the posterior
        
        Returns:
            None
        """
        self.pprogram = pprogram
        self.n_samples = n_samples
        self.n_warmup = n_warmup

        self.nuts_kernel = NUTS(self.pprogram, adapt_step_size=True)
        self.mcmc = MCMC(self.nuts_kernel, num_samples=self.n_samples, warmup_steps=self.n_warmup)


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
        return posterior_samples