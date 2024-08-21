import pyro.distributions
import torch 
import pyro 
import pyro.distributions as dist
try:
        import LM_abstract
        from Quantizer import Quantizer
except:
      from PFNExperiments.LinearRegression.GenerativeModels import LM_abstract 
      from PFNExperiments.LinearRegression.GenerativeModels.Quantizer import Quantizer


def make_lm_program_ig_intercept_batched(
        tau: float = 1.0,
        a: float = 5.0,
        b: float = 2.0,
        tau_beta0: float = 3.0,
    ) -> 'LM_abstract.pprogram_linear_model_return_dict':
        """
        Make a linear model probabilistic program with a gamma prior on sigma_squared and a gamma prior on beta_var.
        Args: 
                tau: float: the standard deviation of the prior on beta
                a: float: the shape parameter of the gamma prior on sigma_squared
                b: float: the rate parameter of the gamma prior on sigma_squared
                tau_beta0: float: the standard deviation of the prior on beta_0
        Returns:
                LM_abstract.pprogram_linear_model_return_dict: a linear model probabilistic program
        """
        def multivariate_lm_return_dict(x: torch.Tensor, y: torch.Tensor = None) -> dict:
                if x.dim() == 2:
                        x = x.unsqueeze(0)  # Ensure x is 3D (batch_size, N, P)
                batch_size, N, P = x.shape

                # Define distributions for the global parameters
                sigma_squared_dist = dist.InverseGamma(a, b)


                beta_cov = torch.eye(P) * (tau ** 2)  # the covariance matrix of the parameters of the linear model

                dist_beta0 = dist.Normal(0, tau_beta0**2)

                with pyro.plate("batch", batch_size, dim=-1):   
                        
                        sigma_squared = pyro.sample("sigma_squared", sigma_squared_dist).squeeze() # Shape: (batch_size,)

                        #print(beta_cov.shape)
                        beta_dist = dist.MultivariateNormal(torch.zeros(P), beta_cov)
                        #print(beta_dist.batch_shape, beta_dist.event_shape)
                        beta = pyro.sample("beta", beta_dist)  # Shape: (batch_size, P)

                
                        # Compute mean using matrix multiplication
                        mean = torch.matmul(x, beta.unsqueeze(-1)).squeeze(-1)  # Shape: (batch_size, N)


                        with pyro.plate("data", N):
                                noise = pyro.sample("noise", dist.Normal(0, sigma_squared))  # Shape: (batch_size, N)

                        beta0 = pyro.sample("beta0", dist_beta0)
                        beta0 = beta0.unsqueeze(-1)  # Shape: (batch_size, 1)
                        
                        noise = noise.permute(1, 0)  # Shape: (N, batch_size)
                        y = mean + noise  # Shape: (batch_size, N)
                        y = y + beta0 # Shape: (batch_size, N)

                

                sigma_squared = sigma_squared.unsqueeze(-1)
                beta = torch.cat([beta0, beta], dim=-1)

                return {
                                "x": x,
                                "y": y,
                                "sigma_squared": sigma_squared,
                                "beta": beta,
                                "beta0": beta0
                        }

        return multivariate_lm_return_dict

def make_lm_program_ig_intercept(
        tau: float = 1.0,
        a: float = 5.0,
        b: float = 2.0,
        tau_beta0: float = 10.0,
    ) -> 'LM_abstract.pprogram_linear_model_return_dict':
        """
        Make a linear model probabilistic program with a gamma prior on sigma_squared and a gamma prior on beta_var.
        This version is not batched!

        Args:
                tau: float: the standard deviation of the prior on beta
                a: float: the shape parameter of the gamma prior on sigma_squared
                b: float: the rate parameter of the gamma prior on sigma_squared
                tau_beta0: float: the standard deviation of the prior on beta_0
        Returns:
                LM_abstract.pprogram_linear_model_return_dict: a linear model probabilistic program
        """

        
        def multivariate_lm_return_dict(x: torch.Tensor, y: torch.Tensor = None) -> dict:
                # Define distributions for the global parameters
                sigma_squared_dist = dist.InverseGamma(a, b)

                beta_cov = torch.eye(x.shape[1]) * (tau ** 2)  # the covariance matrix of the parameters of the linear model

                sigma_squared = pyro.sample("sigma_squared", sigma_squared_dist).squeeze()

                beta_dist = dist.MultivariateNormal(torch.zeros(x.shape[1]), beta_cov)

                dist_beta0 = dist.Normal(0, tau_beta0**2)

                beta = pyro.sample("beta", beta_dist)

                # Compute mean using matrix multiplication
                mean = torch.matmul(x, beta)

                beta_0 = pyro.sample("beta0", dist_beta0)

                mean = mean + beta_0

                with pyro.plate("data", len(x)):
                        y = pyro.sample("obs", pyro.distributions.Normal(mean, sigma_squared), obs=y)


                beta = torch.cat([beta_0.unsqueeze(-1), beta], dim=-1)
                return {
                        "x": x,
                        "y": y,
                        "sigma_squared": sigma_squared,
                        "beta": beta,
                        "beta0": beta_0
                }

        return multivariate_lm_return_dict

