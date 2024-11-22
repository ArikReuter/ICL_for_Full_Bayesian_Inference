# Standard Library Imports
from copy import deepcopy

# Pyro Imports
import pyro
from pyro.infer import SVI, Trace_ELBO, TraceEnum_ELBO
from pyro.infer.autoguide import AutoDelta, AutoGuideList, AutoDiscreteParallel
from pyro.infer.autoguide.initialization import init_to_sample
from pyro.optim import PyroOptim
from pyro.poutine import block

# PyTorch Imports
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.optimizer import Optimizer, required

# Custom Imports
from PFNExperiments.LinearRegression.GenerativeModels.LM_abstract import ppgram_linear_model_return_y
from PFNExperiments.LinearRegression.ComparisonModels.PosteriorComparisonModel import PosteriorComparisonModel




class SGLD_optim(Optimizer):
    """SGLD based on Welling, Max, and Yee W. Teh. "Bayesian learning via stochastic gradient Langevin dynamics." Proceedings of the 28th international conference on machine learning (ICML-11). 2011.
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
    Use SGLD to obtain samples for a given probabilistic program with mini-batch processing.
    """

    def __init__(self, 
                 pprogram: ppgram_linear_model_return_y,
                 n_samples: int = 200,
                 n_warmup: int = 100,
                 n_batches: int = 1,
                 shuffle_samples: bool = True,
                 optim_kwargs: dict = {"lr": 1e-4},
                 discrete_z: bool = False,
                 ) -> None:
        """
        Args:
            pprogram: ppgram_linear_model_return_y: the probabilistic program
            n_samples: int: the number of samples to draw from the posterior
            n_warmup: int: the number of warmup samples to draw from the posterior
            n_batches: int: the number of batches for mini-batch processing
            shuffle_samples: bool: whether to shuffle the samples
            optim_kwargs: dict: optimization hyperparameters
            discrete_z: bool: whether the latent variable z is discrete
        
        Returns:
            None
        """
        self.pprogram = pprogram
        self.n_samples = n_samples
        self.n_warmup = n_warmup
        self.n_batches = n_batches
        self.shuffle_samples = shuffle_samples
        self.optim_kwargs = optim_kwargs
        self.discrete_z = discrete_z

        # Set up the guide
        if not self.discrete_z:
            self.guide = AutoDelta(self.pprogram, init_loc_fn=init_to_sample)
        else:
            self.guide = AutoGuideList(self.pprogram)
            self.guide.append(AutoDelta(block(self.pprogram, hide=["z"])))
            self.guide.append(AutoDiscreteParallel(block(self.pprogram, expose=["z"])))

    def _create_dataloader(self, X: torch.Tensor, y: torch.Tensor) -> DataLoader:
        """
        Create a DataLoader for mini-batch processing.

        Args:
            X: torch.Tensor: covariates
            y: torch.Tensor: response variable
        
        Returns:
            DataLoader: the data loader for mini-batches
        """
        dataset = TensorDataset(X, y)
        return DataLoader(dataset, batch_size=max(len(X) // self.n_batches, 1), shuffle=True)

    def sample_posterior(self,  
                         X: torch.Tensor,
                         y: torch.Tensor) -> dict:
        """
        A method that samples from the posterior distribution with mini-batch processing.

        Args:
            X: torch.Tensor: the covariates
            y: torch.Tensor: the response variable
        
        Returns:
            dict: the samples from the posterior distribution
        """
        dataloader = self._create_dataloader(X, y)

        # Choose the appropriate loss function
        if self.discrete_z:
            loss_fn = TraceEnum_ELBO(max_plate_nesting=10)
        else:
            loss_fn = Trace_ELBO()

        # Set up the optimizer and SVI
        optim = PyroOptim(SGLD_optim, self.optim_kwargs)
        svi = SVI(self.pprogram, self.guide, optim, loss=loss_fn)

        # Warmup phase
        warmup_steps = 0
        while warmup_steps < self.n_warmup:
            for X_batch, y_batch in dataloader:
                if warmup_steps >= self.n_warmup:
                    break
                svi.step(X_batch, y_batch)
                warmup_steps += 1

        # Sampling phase
        samples = []
        collected_samples = 0
        while collected_samples < self.n_samples:
            for X_batch, y_batch in dataloader:
                if collected_samples >= self.n_samples:
                    break
                if not self.discrete_z:
                    samples.append(deepcopy(self.guide.median()))
                else:
                    predictive = pyro.infer.Predictive(self.guide, num_samples=1)
                    sample = predictive(X_batch)
                    samples.append(sample)
                svi.step(X_batch, y_batch)
                collected_samples += 1

        # Convert samples into a structured dictionary
        samples = {k: torch.stack([s[k].squeeze() for s in samples]) for k in samples[0].keys()}
        
        if self.shuffle_samples:
            for k in samples.keys():
                samples[k] = samples[k][torch.randperm(samples[k].shape[0])]

        return samples

    def __repr__(self) -> str:
        return "SGLD"
