import pyro.distributions
import torch 
import pyro 
import pyro.distributions as dist
try:
        import LM_abstract
except:
      from PFNExperiments.LinearRegression.GenerativeModels import LM_abstract 

"""
A class where a linear model with Beta prior on the paramters beta is used
"""

def make_lm_program_beta_prior_batched(
        a: float = 5.0,
        b: float = 2.0,
        alpha_betadist: float = 1.0,
        beta_betadist: float = 1.0
    ) -> 'LM_abstract.pprogram_linear_model_return_dict':
        """
        Make a linear model probabilistic program with a gamma prior on sigma_squared and a gamma prior on beta_var.
        Args: 
                tau: float: the standard deviation of the prior on beta
                a: float: the shape parameter of the gamma prior on sigma_squared
                b: float: the rate parameter of the gamma prior on sigma_squared
                alpha: float: the shape parameter of the gamma prior on beta
        Returns:
                LM_abstract.pprogram_linear_model_return_dict: a linear model probabilistic program
        """
        def multivariate_lm_return_dict(x: torch.Tensor, y: torch.Tensor = None) -> dict:
                if x.dim() == 2:
                        x = x.unsqueeze(0)  # Ensure x is 3D (batch_size, N, P)
                batch_size, N, P = x.shape

                # Define distributions for the global parameters
                sigma_squared_dist = dist.InverseGamma(a, b)


                with pyro.plate("batch", batch_size, dim=-1):   
                        
                        sigma_squared = pyro.sample("sigma_squared", sigma_squared_dist).squeeze() # Shape: (batch_size,)
                        
                        with pyro.plate("beta_plate", P):
                                beta = pyro.sample("beta", dist.Beta(alpha_betadist, beta_betadist))
                        
                        beta = beta.T
                        print(beta.shape)

                        # Compute mean using matrix multiplication
                        mean = torch.matmul(x, beta.unsqueeze(-1)).squeeze(-1)  # Shape: (batch_size, N)


                        with pyro.plate("data", N):
                                noise = pyro.sample("noise", dist.Normal(0, sigma_squared))  # Shape: (batch_size, N)
                        
                        noise = noise.permute(1, 0)  # Shape: (N, batch_size)
                        y = mean + noise  # Shape: (batch_size, N)

                

                sigma_squared = sigma_squared.unsqueeze(-1)

                return {
                                "x": x,
                                "y": y,
                                "sigma_squared": sigma_squared,
                                "beta": beta
                        }

        return multivariate_lm_return_dict

def make_lm_program_beta_prior(
        a: float = 5.0,
        b: float = 2.0,
        alpha_betadist: float = 1.0,
        beta_betadist: float = 1.0
    ) -> 'LM_abstract.pprogram_linear_model_return_dict':
        """
        Make a linear model probabilistic program with a gamma prior on sigma_squared and a gamma prior on beta_var.
        This version is not batched!

        Args:
                tau: float: the standard deviation of the prior on beta
                a: float: the shape parameter of the gamma prior on sigma_squared
                b: float: the rate parameter of the gamma prior on sigma_squared
                alpha: float: the shape parameter of the gamma prior on beta
        Returns:
                LM_abstract.pprogram_linear_model_return_dict: a linear model probabilistic program
        """
        def multivariate_lm_return_dict(x: torch.Tensor, y: torch.Tensor = None) -> dict:
                # Define distributions for the global parameters
                sigma_squared_dist = dist.InverseGamma(a, b)


                sigma_squared = pyro.sample("sigma_squared", sigma_squared_dist).squeeze()


                with pyro.plate("beta_plate", x.shape[1]):
                        beta = pyro.sample("beta", dist.Beta(alpha_betadist, beta_betadist))
                
                # Compute mean using matrix multiplication
                mean = torch.matmul(x, beta)

                #print(f"beta: {beta.shape}")
                #print(f"x: {x.shape}")
                #print(f"mean: {mean.shape}")
                #print(f"sigma_squared: {sigma_squared.shape}")

                with pyro.plate("data", len(x)):
                        y = pyro.sample("obs", pyro.distributions.Normal(mean, sigma_squared), obs=y)


                return {
                        "x": x,
                        "y": y,
                        "sigma_squared": sigma_squared,
                        "beta": beta
                }

        return multivariate_lm_return_dict

