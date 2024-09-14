from numpyro.infer import MCMC, NUTS
import torch
from PFNExperiments.LinearRegression.ComparisonModels.PosteriorComparisonModel import PosteriorComparisonModel
from jax import random
import jax.numpy as jnp
import numpy as np

class Hamiltionian_MC(PosteriorComparisonModel):
    """
    use Hamiltonian Monte Carlo to obtain posterior samples for a given probabilistic program
    """

    def __init__(self, 
                 pprogram,
                 n_samples:int = 200,
                 n_warmup:int = 100,
                 kernel_kwargs: dict = {},
                 mcmc_kwargs: dict = {},
                 shuffle_samples: bool = True,
                 sample_key = None

                 ) -> None:
        
        """
        Args:
            pprogram:
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
        self.mcmc = MCMC(self.nuts_kernel, num_samples=self.n_samples, num_warmup=self.n_warmup, **mcmc_kwargs)
        self.shuffle_samples = shuffle_samples

        if sample_key is None:
            sample_key = random.PRNGKey(0)
            rng_key, key = random.split(sample_key)

            self.sample_key = key


    def sample_posterior(self,  
                x: torch.tensor,
                y = None) -> torch.Tensor:
        """
        A method that samples from the posterior distribution
        Args:
            X: torch.Tensor: the observed variables
        Returns:
            torch.Tensor: the samples from the posterior distribution
        """
        x = jnp.asarray(x)
        self.mcmc.run(rng_key=self.sample_key, x=x)
        posterior_samples = self.mcmc.get_samples()

        posterior_samples = {k: torch.tensor(np.asarray(v)) for k, v in posterior_samples.items()}
        if self.shuffle_samples:
            posterior_samples = {k: v[torch.randperm(v.shape[0])] for k, v in posterior_samples.items()}

        return posterior_samples
    

    def __repr__(self) -> str:
        return "Hamiltonian Monte Carlo Numpyro"