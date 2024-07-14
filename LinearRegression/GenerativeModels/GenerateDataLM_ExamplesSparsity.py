import pyro.distributions
import torch 
import pyro 
import pyro.distributions as dist
try:
        import LM_abstract
except:
      from PFNExperiments.LinearRegression.GenerativeModels import LM_abstract 



def make_lm_program_Laplace_ig_batched(
        tau: float = 1.0,
        a: float = 5.0,
        b: float = 2.0
    ) -> 'LM_abstract.pprogram_linear_model_return_dict':
        """
        Make a linear model probabilistic program with an inverse gamma prior on sigma_squared and a Laplace prior on beta.
        Args: 
                tau: float: the standard deviation of the prior on beta
                a: float: the shape parameter of the gamma prior on sigma_squared
                b: float: the rate parameter of the gamma prior on sigma_squared
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

                        #print(beta_cov.shape)
                        beta_dist = dist.Laplace(torch.zeros(P, 1), torch.ones(P, 1) * tau)  # Shape: (batch_size, P)
                        #print(beta_dist.batch_shape, beta_dist.event_shape)
                        beta = pyro.sample("beta", beta_dist).T  # Shape: (batch_size, P)
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

def make_lm_program_Laplace_ig(
        tau: float = 1.0,
        a: float = 5.0,
        b: float = 2.0
    ) -> 'LM_abstract.pprogram_linear_model_return_dict':
        """
        Make a linear model probabilistic program with a gamma prior on sigma_squared and a gamma prior on beta_var.
        This version is not batched!

        Args:
                tau: float: the standard deviation of the prior on beta
                a: float: the shape parameter of the gamma prior on sigma_squared
                b: float: the rate parameter of the gamma prior on sigma_squared
        Returns:
                LM_abstract.pprogram_linear_model_return_dict: a linear model probabilistic program
        """
        def multivariate_lm_return_dict(x: torch.Tensor, y: torch.Tensor = None) -> dict:
                # Define distributions for the global parameters
                sigma_squared_dist = dist.InverseGamma(a, b)


                sigma_squared = pyro.sample("sigma_squared", sigma_squared_dist).squeeze()

                beta_dist = dist.Laplace(torch.zeros(x.shape[1]), torch.ones(x.shape[1]) * tau)
                beta = pyro.sample("beta", beta_dist)

                # Compute mean using matrix multiplication
                mean = torch.matmul(x, beta)

                with pyro.plate("data", len(x)):
                        y = pyro.sample("obs", pyro.distributions.Normal(mean, sigma_squared), obs=y)


                return {
                        "x": x,
                        "y": y,
                        "sigma_squared": sigma_squared,
                        "beta": beta
                }

        return multivariate_lm_return_dict