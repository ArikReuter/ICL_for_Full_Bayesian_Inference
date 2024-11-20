import pyro 
from pyro.infer.autoguide import AutoDelta
from pyro.infer import SVI, Trace_ELBO
import torch

from pyro.infer.autoguide.initialization import init_to_sample

from PFNExperiments.LinearRegression.GenerativeModels.LM_abstract import ppgram_linear_model_return_y
from PFNExperiments.LinearRegression.ComparisonModels.PosteriorComparisonModel import PosteriorComparisonModel

from pyro.optim import PyroOptim
from torch.optim.optimizer import Optimizer, required
from copy import deepcopy


class SGLD_optim(Optimizer):
    """Implements SGLD algorithm based on
        https://www.ics.uci.edu/~welling/publications/papers/stoclangevin_v6.pdf

    Built on the PyTorch SGD implementation
    (https://github.com/pytorch/pytorch/blob/v1.4.0/torch/optim/sgd.py)
    """

    def __init__(self,
                 params,
                 lr=required,
                 momentum=0,
                 dampening=0,
                 weight_decay=0,
                 nesterov=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError(
                "Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr,
                        momentum=momentum,
                        dampening=dampening,
                        weight_decay=weight_decay,
                        nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError(
                "Nesterov momentum requires a momentum and zero dampening")
        super(SGLD_optim, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGLD_optim, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(p.data, alpha=weight_decay)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(
                            d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                p.data.add_(d_p, alpha=-group['lr'])
                noise_std = (2 * group['lr']) ** 0.5  # A scalar value
                noise = p.data.new(p.data.size()).normal_(mean=0, std=noise_std)
                p.data.add_(noise)

        return 1.0


class SGLD(PosteriorComparisonModel):
    """
    use SGLD to obtain samples for a given probabilistic program
    """

    def __init__(self, 
                 pprogram: ppgram_linear_model_return_y,
                 n_samples:int = 200,
                 n_warmup:int = 100,
                 n_batches:int = 1,
                 shuffle_samples: bool = True,
                 optim_kwargs: dict = {"lr": 1e-4}
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
        self.shuffle_samples = shuffle_samples
        self.optim_kwargs = optim_kwargs
        self.n_batches = n_batches

        self.guide = AutoDelta(self.pprogram, init_loc_fn=init_to_sample)



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

        Trace_ELBO().differentiable_loss(self.pprogram, self.guide, X, y)

        optim = PyroOptim(SGLD_optim, self.optim_kwargs)

        svi = SVI(self.pprogram, self.guide, optim, loss=Trace_ELBO())

        for i in range(self.n_warmup):
            loss = svi.step(X, y)

        samples = []

        for i in range(self.n_samples):
            samples.append(deepcopy(self.guide.median()))
            svi.step(X, y)

        samples = {k: torch.stack([s[k] for s in samples]) for k in samples[0].keys()}
    
        if self.shuffle_samples:
            for k in samples.keys():
                samples[k] = samples[k][torch.randperm(samples[k].shape[0])]

        return samples

    def __repr__(self) -> str:
        return "SGLD"