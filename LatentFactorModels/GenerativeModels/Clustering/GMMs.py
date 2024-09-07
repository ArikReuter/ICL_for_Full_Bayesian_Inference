
import pyro.distributions
import torch 
import pyro 
import pyro.distributions as dist
from copy import deepcopy

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

    p = int(p)
    n = int(n)
    
    def gmm_program_univariate(x: torch.tensor = None) -> dict:
        """
        A univariate GMM program

        Args:
            x: torch.tensor: the input

        Returns:
            torch.tensor: the output
        """
        #if x is not None:
        #    n = x.shape[0]
        #    p = x.shape[1]  
        
        if x is not None:
            x = x.squeeze()

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

    def gmm_program_univariate(x: torch.tensor = None, y: torch.tensor = None, n = n, p = p, batch_size = batch_size) -> dict:
        """
        A univariate GMM program

        Args:
            x: torch.tensor: the input

        Returns:
            torch.tensor: the output
        """
        
        if x is not None:
            n = x.shape[1]
            p = x.shape[2]
            batch_size = x.shape[0]
        
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

            # make the batch dimension the first dimension
            #phi = phi.transpose(0, 1)
            mu = mu.transpose(0, 1).float()
            sigma_squared = sigma_squared.transpose(0, 1).float()
            z = z.transpose(0, 1).float()
            x = x.transpose(0, 1).float().unsqueeze(-1)

            beta = torch.cat([mu, sigma_squared], dim=1).float()



        return {
            "phi": phi.float(),
            "mu": mu,
            "sigma_squared": sigma_squared,
            "z": z,
            "x": x,
            "beta": beta
        }
    
    return gmm_program_univariate


def make_gmm_program_spherical(
        n: int = 100,
        p: int = 5,
        k: int = 3,
        a1: float = 5.0,
        b1: float = 2.0,
        dirichlet_beta: float = 1.0,
        lambda1: float = 3.0,
):
    """
    Make a probabilistic program for a univariate GMM 
    Args:   
        N: int: number of data points
        p: int: number of dimensions
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

    p = int(p)
    n = int(n)
    k = int(k)
    
    def gmm_program(x: torch.tensor = None, p = p, k = k, n = n) -> dict:
        """
        A univariate GMM program

        Args:
            x: torch.tensor: the input

        Returns:
            torch.tensor: the output
        """
        #if x is not None:
        #    n = x.shape[0]
        #    p = x.shape[1]  
        
        if x is not None:
            x = x.squeeze()
        
        k = int(k)

        sigma_squared_dist = pyro.distributions.InverseGamma(a1, b1)

        phi_dist = pyro.distributions.Dirichlet(torch.tensor([dirichlet_beta] * k))
        phi = pyro.sample("phi", phi_dist) # mixture weights

        with pyro.plate("components", k):
            sigma_squared = pyro.sample("sigma_squared", sigma_squared_dist)
            mu = pyro.sample("mu", dist.Normal(0, lambda1 * sigma_squared))

        with pyro.plate("data", n):
            z = pyro.sample("z", dist.Categorical(phi))
            mu_z = mu[z]
            sigma_squared_z = sigma_squared[z]
            
            with pyro.plate("dims", p):
                x = pyro.sample("x", dist.Normal(mu_z, sigma_squared_z), obs=x)
        
        #x = x.transpose(0, 1)
        return {
            "phi": phi,
            "mu": mu,
            "sigma_squared": sigma_squared,
            "z": z,
            "x": x,
        }
    
    return gmm_program



def make_gmm_program_spherical_batched(
        n: int = 100,
        p: int = 5,
        k: int = 3, 
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
        p: int: number of dimensions
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

    def gmm_program_shperical(x: torch.tensor = None, y: torch.tensor = None, n = n, p = p, batch_size = batch_size, k = k) -> dict:
        """
        A univariate GMM program

        Args:
            x: torch.tensor: the input

        Returns:
            torch.tensor: the output
        """
        
        if x is not None:
            n = x.shape[1]
            p = x.shape[2]
            batch_size = x.shape[0]
        
        k = int(k)
        
        sigma_squared_dist = pyro.distributions.InverseGamma(a1, b1)

        phi_dist = pyro.distributions.Dirichlet(torch.tensor([dirichlet_beta] * k))

        with pyro.plate("batch", batch_size):
            phi = pyro.sample("phi", phi_dist) # mixture weights

            with pyro.plate("dims", k):
                sigma_squared = pyro.sample("sigma_squared", sigma_squared_dist)
                mu = pyro.sample("mu", dist.Normal(0, lambda1 * sigma_squared))

            with pyro.plate("data", n):
                z = pyro.sample("z", dist.Categorical(phi))
                mu_z = mu[z, torch.arange(batch_size).expand(n, batch_size)]
                sigma_squared_z = sigma_squared[z, torch.arange(batch_size).expand(n, batch_size)]

                with pyro.plate("components", p):
                    x = pyro.sample("x", dist.Normal(mu_z, sigma_squared_z))

            


            # make the batch dimension the first dimension
            #phi = phi.transpose(0, 1)
            mu = mu.transpose(0, 1).float()
            sigma_squared = sigma_squared.transpose(0, 1).float()
            z = z.transpose(0, 1).float()

            beta = torch.cat([mu, sigma_squared], dim=1).float()



        return {
            "phi": phi.float(),
            "mu": mu,
            "sigma_squared": sigma_squared,
            "z": z,
            "x": x,
            "beta": beta
        }
    
    return gmm_program_shperical



def make_gmm_program_diagonal(
        n: int = 100,
        p: int = 5,
        k: int = 3,
        a1: float = 5.0,
        b1: float = 2.0,
        dirichlet_beta: float = 1.0,
        lambda1: float = 3.0,
):
    """
    Make a probabilistic program for a univariate GMM 
    Args:   
        N: int: number of data points
        p: int: number of dimensions
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

    p = int(p)
    n = int(n)
    k = int(k)
    
    def gmm_program(x: torch.tensor = None, p = p, k = k, n = n) -> dict:
        """
        A univariate GMM program

        Args:
            x: torch.tensor: the input

        Returns:
            torch.tensor: the output
        """
        #if x is not None:
        #    n = x.shape[0]
        #    p = x.shape[1]  
        
        if x is not None:
            x = x.squeeze()
        
        k = int(k)

        sigma_squared_dist = pyro.distributions.InverseGamma(a1, b1)

        phi_dist = pyro.distributions.Dirichlet(torch.tensor([dirichlet_beta] * k))
        phi = pyro.sample("phi", phi_dist) # mixture weights
    	
        with pyro.plate("dims", p):
            with pyro.plate("components", k):
                sigma_squared = pyro.sample("sigma_squared", sigma_squared_dist)
                mu = pyro.sample("mu", dist.Normal(0, lambda1 * sigma_squared))
        
        with pyro.plate("data", n):
            z = pyro.sample("z", dist.Categorical(phi))
            mu_z = mu[z]
            sigma_squared_z = sigma_squared[z]

            cov_mat = torch.diag_embed(sigma_squared_z)  
            x = pyro.sample("x", dist.MultivariateNormal(mu_z, cov_mat), obs=x)
        #print(x.shape)
        #x = x.transpose(0, 1)
        return {
            "phi": phi,
            "mu": mu,
            "sigma_squared": sigma_squared,
            "z": z,
            "x": x,
        }
    
    return gmm_program


def make_gmm_program_diagonal_batched(
        n: int = 100,
        p: int = 5,
        k: int = 3,
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
        p: int: number of dimensions
        K: int: number of components
        batch_size: int: the batch size
        a1: float: shape of inverse gamma prior on sigma_squared
        b1: float: scale of inverse gamma prior on sigma_squared
        dirichlet_beta: float: concentration parameter of the Dirichlet prior on the mixture weights
        lambda1: float: scale the variance for the prior on the mus
    """
    a1 = torch.tensor(a1)
    b1 = torch.tensor(b1)
    dirichlet_beta = torch.tensor(dirichlet_beta)
    lambda1 = torch.tensor(lambda1)

    p = int(p)
    n = int(n)
    k = int(k)
    
    def gmm_program(x: torch.tensor = None, p = p, k = k, n = n, batch_size = batch_size) -> dict:
        """
        A univariate GMM program

        Args:
            x: torch.tensor: the input

        Returns:
            torch.tensor: the output
        """
        if x is not None:
            n = x.shape[1]
            p = x.shape[2]
            batch_size = x.shape[0]
        
        if x is not None:
            x = x.squeeze()
        
        k = int(k)

        sigma_squared_dist = pyro.distributions.InverseGamma(a1, b1)

        phi_dist = pyro.distributions.Dirichlet(torch.tensor([dirichlet_beta] * k))
        
    	

        with pyro.plate("batch", batch_size):
            phi = pyro.sample("phi", phi_dist) # mixture weights
            
            with pyro.plate("dims", p):
                with pyro.plate("components", k):
                    sigma_squared = pyro.sample("sigma_squared", sigma_squared_dist)
                    mu = pyro.sample("mu", dist.Normal(0, lambda1 * sigma_squared))
            
            with pyro.plate("data", n):
                z = pyro.sample("z", dist.Categorical(phi))

                batch_size, seq_len = z.shape  # (10, 1024)
                _, num_features, _ = mu.shape  # (3, 4, 1024)

                # We need to gather from mu using z as index along the first dimension (size 3).
                # To do this correctly, we need to match the shapes of the tensors appropriately.

                # Gather mu based on z values along the first dimension
                mu_z = torch.stack([mu[z[:, i], :, i] for i in range(seq_len)], dim=-1)
                sigma_squared_z = torch.stack([sigma_squared[z[:, i], :, i] for i in range(seq_len)], dim=-1)

                cov_mat = torch.diag_embed(sigma_squared_z.permute(2, 0, 1)) 
            
                mu_z = mu_z.permute(0, 2, 1)
                cov_mat = cov_mat.permute(1, 0, 2, 3)

                x = pyro.sample("x", dist.MultivariateNormal(mu_z, cov_mat))

       
        x = x.permute(1, 0, 2)
        mu = mu.permute(2, 0, 1)
        sigma_squared = sigma_squared.permute(2, 0, 1)
        z = z.transpose(0, 1).float()

        mu = mu.flatten(1)
        sigma_squared = sigma_squared.flatten(1)

        beta = torch.cat([mu, sigma_squared], dim=1).float()

        return {
            "phi": phi.float(),
            "mu": mu,
            "sigma_squared": sigma_squared,
            "z": z,
            "x": x,
            "beta": beta
        }
    
    return gmm_program