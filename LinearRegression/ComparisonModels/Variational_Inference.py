import pyro 
from torch.distributions import constraints
import torch

import torch
import pyro
import pyro.infer
import pyro.optim
from pyro.infer import SVI, Trace_ELBO

from PFNExperiments.LinearRegression.GenerativeModels.LM_abstract import ppgram_linear_model_return_y, pprogram_linear_model_return_dict
from PFNExperiments.LinearRegression.GenerativeModels import LM_abstract
from PFNExperiments.LinearRegression.ComparisonModels.PosteriorComparisonModel import PosteriorComparisonModel

def make_guide_program_gamma_gamma(a0:float = 5.0,
        b0: float = 2.0,
        a1: float = 5.0,
        b1: float = 2.0
        ) -> pprogram_linear_model_return_dict:
    
    """
    Make a guide program for a linear model with a gamma prior on sigma_squared and a gamma prior on beta_var.
    Args:
        a0: float: the shape parameter of the gamma prior on beta_var
        bo: float: the rate parameter of the gamma prior on beta_var
        a1: float: the shape parameter of the gamma prior on sigma_squared
        b1: float: the rate parameter of the gamma prior on sigma_squared
    Returns:
        LM_abstract.pprogram_linear_model_return_dict: a linear model probabilistic program

    """

    def guide_multivariate_lm(x, y=None):
        # Learnable parameters for beta_var's Gamma distribution
        a0_q = pyro.param("a0_q", torch.tensor(a0), constraint=constraints.positive)
        b0_q = pyro.param("b0_q", torch.tensor(b0), constraint=constraints.positive)
        beta_var_q = pyro.sample("beta_var", pyro.distributions.Gamma(a0_q, b0_q))
        
        # Learnable parameters for sigma_squared's Gamma distribution
        a1_q = pyro.param("a1_q", torch.tensor(a1), constraint=constraints.positive)
        b1_q = pyro.param("b1_q", torch.tensor(b1), constraint=constraints.positive)
        sigma_squared_q = pyro.sample("sigma_squared", pyro.distributions.Gamma(a1_q, b1_q))
        
        # Learnable parameters for beta's Multivariate Normal distribution
        beta_loc_q = pyro.param("beta_loc_q", torch.zeros(x.shape[1]))
        beta_scale_q = pyro.param("beta_scale_q", torch.ones(x.shape[1]), constraint=constraints.positive)
        beta_cov_q = torch.diag(beta_scale_q.pow(2))
        beta_q = pyro.sample("beta", pyro.distributions.MultivariateNormal(beta_loc_q, beta_cov_q))

    return guide_multivariate_lm


class Variational_Inference(PosteriorComparisonModel):
    """
    Perform variational inference on a given probabilistic program
    """

    def __init__(self, 
                 pprogram: ppgram_linear_model_return_y,
                 guide: pprogram_linear_model_return_dict = None,
                 n_steps:int = 2000,
                 n_samples:int = 200,
                 lr: float = 1e-3) -> None:
        """
        Args:
            pprogram: ppgram_linear_model_return_y: the probabilistic program
            guide: ppgram_linear_model_return_dict: the guide program
            n_steps: int: the number of steps to take in the optimization
            lr: float: the learning rate of the optimizer
        Returns:
            None
        """
        self.pprogram = pprogram
        self.guide = guide
        self.n_steps = n_steps
        self.n_samples = n_samples
        self.lr = lr


        if self.guide is None:
            self.generate_autoguide_normal()


        self.optimizer = pyro.optim.Adam({"lr": self.lr})

        self.svi = SVI(self.pprogram, self.guide, self.optimizer, loss=Trace_ELBO())



    def generate_autoguide_normal(self):
        """
        If no guide is provided, generate an autoguide for the model
        """

        self.guide = pyro.infer.autoguide.AutoDiagonalNormal(self.pprogram)


    def do_inference(self,  
                X: torch.Tensor,
                y: torch.Tensor) -> torch.Tensor:
        """
        A method that performs variational inference
        Args:
            X: torch.Tensor: the covariates
            y: torch.Tensor: the response variable
        Returns:
            torch.Tensor: the samples from the posterior distribution
        """
        pyro.clear_param_store()
        for step in range(self.n_steps):
            self.loss = self.svi.step(X, y)
            if step % 100 == 0:
                print('.', end='')
        print()
        return self.loss
    
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
        self.do_inference(X, y)
        posterior_samples = pyro.infer.Predictive(self.guide, num_samples=self.n_samples)(X, y)
        return posterior_samples
    
    def __repr__(self) -> str:
        return "Variational Inference with guide: {}".format(self.guide)