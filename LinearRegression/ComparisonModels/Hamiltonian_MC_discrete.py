import pyro
from pyro.infer.mcmc import MCMC, HMC
from pyro.infer.mcmc.api import TraceEnum_MCMC
from pyro.infer import Predictive
import torch

# Assuming other imports and class dependencies are already correctly included

from PFNExperiments.LinearRegression.GenerativeModels.LM_abstract import ppgram_linear_model_return_y
from PFNExperiments.LinearRegression.ComparisonModels.PosteriorComparisonModel import PosteriorComparisonModel


class Hamiltonian_MC_With_Discrete(PosteriorComparisonModel):
    """
    Use MCMC with support for discrete variables to obtain posterior samples for a given probabilistic program
    """
    def __init__(self, 
                 pprogram: ppgram_linear_model_return_y,
                 n_samples: int = 200,
                 n_warmup: int = 100,
                 max_plate_nesting: int = 1,  # Set according to your model's structure
                 mcmc_kwargs: dict = {}
                 ) -> None:
        self.pprogram = pprogram
        self.n_samples = n_samples
        self.n_warmup = n_warmup

        # Using HMC with options to handle discrete variables through enumeration
        hmc_kernel = HMC(self.pprogram, full_mass=True)  # You may need to adjust kernel parameters based on your model
        self.mcmc = TraceEnum_MCMC(hmc_kernel, num_samples=self.n_samples, warmup_steps=self.n_warmup, max_plate_nesting=max_plate_nesting, **mcmc_kwargs)

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
