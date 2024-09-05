import pyro 
import torch
import os

import pyro.infer
import pyro.optim
from pyro.infer import SVI, Trace_ELBO, TraceEnum_ELBO, config_enumerate, infer_discrete

from PFNExperiments.LinearRegression.GenerativeModels.LM_abstract import ppgram_linear_model_return_y
from PFNExperiments.LinearRegression.ComparisonModels.PosteriorComparisonModel import PosteriorComparisonModel

from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.poutine import block

from pyro.infer.autoguide import AutoGuideList, AutoDiagonalNormal, AutoDiscreteParallel

class Variational_InferenceAutoguide(PosteriorComparisonModel):
    """
    Perform variational inference on a given probabilistic program
    """

    def __init__(self, 
                 pprogram,
                 make_guide_fun: callable = AutoDiagonalNormal,
                 additional_make_guide_args: dict = {},
                 n_steps:int = 2000,
                 n_samples:int = 200,
                 lr: float = 1e-3) -> None:
        """
        Args:
            pprogram
            make_guide_fun : callable: a function that generates the guide
            n_steps: int: the number of steps to take in the optimization
            lr: float: the learning rate of the optimizer

        Returns:
            None
        """
        self.pprogram = pprogram
        self.make_guide_fun = make_guide_fun
        self.additional_make_guide_args = additional_make_guide_args
        self.n_steps = n_steps
        self.n_samples = n_samples
        self.lr = lr
    

        self.generate_guide()
        
        self.optimizer = pyro.optim.Adam({"lr": self.lr})

        elbo = TraceEnum_ELBO(max_plate_nesting=2)

        self.svi = SVI(self.pprogram, self.guide, self.optimizer, loss=elbo)



    def generate_guide(self):
        """
        generate the guide
        """

        self.guide = AutoGuideList(self.pprogram)
        self.guide.append(self.make_guide_fun(block(self.pprogram, hide = ["z"]), **self.additional_make_guide_args))
        self.guide.append(AutoDiscreteParallel(block(self.pprogram, expose = ["z"])))
        self.guide = config_enumerate(self.guide, "parallel")

    def do_inference(self,  
                x: torch.Tensor,
                y = None) -> torch.Tensor:
        """
        A method that performs variational inference
        Args:
            X: torch.Tensor: the data
        Returns:
            torch.Tensor: the samples from the posterior distribution
        """
        pyro.clear_param_store()
        for step in range(self.n_steps):
            self.loss = self.svi.step(x)
            if step % 100 == 0:
                print('.', end='')
        print()
        return self.loss
    
    def sample_posterior(self,
                x: torch.Tensor,
                y = None
                ) -> torch.Tensor:
        """
        A method that samples from the posterior distribution
        Args:
            x: torch.Tensor: the data
        Returns:
            torch.Tensor: the samples from the posterior distribution
        """
        self.do_inference(x)
        posterior_samples = pyro.infer.Predictive(self.guide, num_samples=self.n_samples)(x)
        return posterior_samples
    
    def __repr__(self) -> str:
        return "Variational Inference with guide: {}".format(self.guide)