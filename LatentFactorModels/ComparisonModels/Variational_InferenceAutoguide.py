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
                 discrete_z: bool = True,
                 lr: float = 1e-3,
                 print_lr: bool = False) -> None:
        """
        Args:
            pprogram
            make_guide_fun : callable: a function that generates the guide
            n_steps: int: the number of steps to take in the optimization
            discrete_z: bool: whether the latent variable is discrete
            lr: float: the learning rate of the optimizer
            print_lr: bool: whether to print the learning rate


        Returns:
            None
        """
        self.pprogram = pprogram
        self.make_guide_fun = make_guide_fun
        self.additional_make_guide_args = additional_make_guide_args
        self.n_steps = n_steps
        self.n_samples = n_samples
        self.lr = lr
        self.print_lr = print_lr
    
        self.discrete_z = discrete_z
        self.generate_guide()
        
        self.optimizer = pyro.optim.Adam({"lr": self.lr})

        if self.discrete_z:
            elbo = TraceEnum_ELBO(max_plate_nesting=2)
        else:
            elbo = Trace_ELBO()

        self.elbo = elbo

        self.svi = SVI(self.pprogram, self.guide, self.optimizer, loss=elbo)



    def generate_guide(self):
        """
        generate the guide
        """ 
        if self.discrete_z:
            self.guide = AutoGuideList(self.pprogram)
            self.guide.append(self.make_guide_fun(block(self.pprogram, hide = ["z"]), **self.additional_make_guide_args))
            self.guide.append(AutoDiscreteParallel(block(self.pprogram, expose = ["z"])))
            self.guide = config_enumerate(self.guide, "parallel")
        else:   
            self.guide = self.make_guide_fun(self.pprogram, **self.additional_make_guide_args)


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


        self.svi = SVI(self.pprogram, self.guide, self.optimizer, loss=self.elbo)
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
        try:
            posterior_samples = pyro.infer.Predictive(self.guide, num_samples=self.n_samples)(x)

        except ValueError as e:
            print(e)
            posterior_samples = None
        return posterior_samples
    
    def __repr__(self) -> str:
        rep =  "Variational Inference with guide: {}".format(self.guide)

        if self.print_lr:
            rep = f"Variational Inference with lr: {self.lr} and guide: {self.guide}"

        return rep