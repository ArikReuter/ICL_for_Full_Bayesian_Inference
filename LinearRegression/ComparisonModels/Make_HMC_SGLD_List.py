from PFNExperiments.LinearRegression.ComparisonModels.Hamiltionion_MC import Hamiltionian_MC
from PFNExperiments.LinearRegression.ComparisonModels.SGLD import SGLD


def make_hmc_sgld_list(
        pprogram_y,
        n_samples: int = 1000,
        ):
    """
    Create a model list with HMC and SGLD
    Args:
        pprogram_y: a probabilistic program for the response variable
        n_samples: int: the number of samples to draw
    """

    hmc_sampler = Hamiltionian_MC(pprogram=pprogram_y, n_warmup=n_samples//2, n_samples=n_samples)

    sgld_sampler = SGLD(pprogram=pprogram_y, n_warmup = 10_000, n_samples = n_samples, optim_kwargs = {"lr": 1e-4})

    model_list = [
        hmc_sampler,
        sgld_sampler
    ]

    return model_list

  