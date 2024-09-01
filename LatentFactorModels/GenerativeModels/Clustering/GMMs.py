
import pyro.distributions
import torch 
import pyro 
import pyro.distributions as dist

def make_gmm_program_univariate(
        n: int = 100,
        p: int = 3,
        a1: float = 5.0,
        b1: float = 2.0,
        dirichlet_beta: float = 1.0,
        lambda1: float = 3.0,
):
    """
    Make a probabilistic program for a univariate GMM 
    Args:   
        N: int: number of data points
        K: int: number of components
        a1: float: shape of inverse gamma prior on sigma_squared
        b1: float: scale of inverse gamma prior on sigma_squared
        dirichlet_beta: float: concentration parameter of the Dirichlet prior on the mixture weights
        lambda1: float: scale the variance for the prior on the mus
    """
    a1 = torch.tensor(a1)
    b1 = torch.tensor(b1)
    dirichlet_beta = torch.tensor(dirichlet_beta)
    lambda1 = torch.tensor(lambda1)
    
    def gmm_program_univariate(x: torch.tensor = None) -> dict:
        """
        A univariate GMM program

        Args:
            x: torch.tensor: the input

        Returns:
            torch.tensor: the output
        """
        sigma_squared_dist = pyro.distributions.InverseGamma(a1, b1)

        phi_dist = pyro.distributions.Dirichlet(torch.tensor([dirichlet_beta] * p))
        phi = pyro.sample("phi", phi_dist) # mixture weights

        with pyro.plate("components", p):
            sigma_squared = pyro.sample("sigma_squared", sigma_squared_dist)
            mu = pyro.sample("mu", dist.Normal(0, lambda1 * sigma_squared))

        with pyro.plate("data", n):
            z = pyro.sample("z", dist.Categorical(phi))
            x = pyro.sample("x", dist.Normal(mu[z], sigma_squared[z]), obs=x)

        return {
            "phi": phi,
            "mu": mu,
            "sigma_squared": sigma_squared,
            "z": z,
            "x": x,
        }
    
    return gmm_program_univariate


def make_gmm_program_univariate_batched(
        n: int = 100,
        p: int = 3,
        batch_size: int = 1000,
        a1: float = 5.0,
        b1: float = 2.0,
        dirichlet_beta: float = 1.0,
        lambda1: float = 3.0,
):
    """
    Make a probabilistic program for a univariate GMM 
    Args:   
        N: int: number of data points
        K: int: number of components
        a1: float: shape of inverse gamma prior on sigma_squared
        b1: float: scale of inverse gamma prior on sigma_squared
        dirichlet_beta: float: concentration parameter of the Dirichlet prior on the mixture weights
        lambda1: float: scale the variance for the prior on the mus
    """
    a1 = torch.tensor(a1)
    b1 = torch.tensor(b1)
    dirichlet_beta = torch.tensor(dirichlet_beta)
    lambda1 = torch.tensor(lambda1)

    def gmm_program_univariate(x: torch.tensor = None) -> dict:
        """
        A univariate GMM program

        Args:
            x: torch.tensor: the input

        Returns:
            torch.tensor: the output
        """
        if x is not None:
            assert x.shape[0] == batch_size, "The batch size of the input must be equal to the batch size of the model"
            assert x.shape[1] == n, "The number of data points must be equal to the number of data points in the model"
            assert x.shape[2] == p, "The number of features must be equal to 1"


        sigma_squared_dist = pyro.distributions.InverseGamma(a1, b1)

        phi_dist = pyro.distributions.Dirichlet(torch.tensor([dirichlet_beta] * p))

        with pyro.plate("batch", batch_size):
            phi = pyro.sample("phi", phi_dist) # mixture weights

            with pyro.plate("components", p):
                sigma_squared = pyro.sample("sigma_squared", sigma_squared_dist)
                mu = pyro.sample("mu", dist.Normal(0, lambda1 * sigma_squared))

            with pyro.plate("data", n):
                z = pyro.sample("z", dist.Categorical(phi))
                mu_z = mu[z, torch.arange(batch_size).expand(n, batch_size)]
                sigma_squared_z = sigma_squared[z, torch.arange(batch_size).expand(n, batch_size)]

            x = pyro.sample("x", dist.Normal(mu_z, sigma_squared_z))

        return {
            "phi": phi,
            "mu": mu,
            "sigma_squared": sigma_squared,
            "z": z,
            "x": x,
        }
    
    return gmm_program_univariate