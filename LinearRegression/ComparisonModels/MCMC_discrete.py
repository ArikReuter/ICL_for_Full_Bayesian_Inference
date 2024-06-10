import pyro
from pyro.infer import MCMC, NUTS, config_enumerate
from pyro.infer import Predictive
import torch


# Assuming other imports and class dependencies are already correctly included
from PFNExperiments.LinearRegression.GenerativeModels.LM_abstract import ppgram_linear_model_return_y
from PFNExperiments.LinearRegression.ComparisonModels.PosteriorComparisonModel import PosteriorComparisonModel

class MCMC_discrete(PosteriorComparisonModel):
    """
    Use MCMC with support for discrete variables to obtain posterior samples for a given probabilistic program.
    This class configures the probabilistic program to handle discrete variables by using enumeration.
    """
    def __init__(self, 
                 pprogram: ppgram_linear_model_return_y,
                 n_samples: int = 200,
                 n_warmup: int = 100,
                 max_plate_nesting: int = 1,  # Adjust according to your model's structure
                 mcmc_kwargs: dict = {}
                 ) -> None:
        # Decorate the probabilistic program with enumeration to handle discrete variables
        self.pprogram = config_enumerate(pprogram, "parallel")

        self.n_samples = n_samples
        self.n_warmup = n_warmup

        # Configure NUTS kernel for MCMC
        nuts_kernel = NUTS(self.pprogram, max_plate_nesting=max_plate_nesting)
        self.mcmc = MCMC(nuts_kernel, num_samples=self.n_samples, warmup_steps=self.n_warmup, **mcmc_kwargs)

    def sample_posterior(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Samples from the posterior distribution using configured MCMC.
        Args:
            X: torch.Tensor: the covariates
            y: torch.Tensor: the response variable
        Returns:
            torch.Tensor: the samples from the posterior distribution
        """
        self.mcmc.run(X, y)
        posterior_samples = self.mcmc.get_samples()
        return posterior_samples
