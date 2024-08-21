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
                
                        # Compute mean using matrix multiplication
                        mean = torch.matmul(x, beta.unsqueeze(-1)).squeeze(-1)  # Shape: (batch_size, N)


                        with pyro.plate("data", N):
                                noise = pyro.sample("noise", dist.Normal(0, sigma_squared))  # Shape: (batch_size, N)
                        
                        noise = noise.permute(1, 0)  # Shape: (N, batch_size)
                        y = mean + noise  # Shape: (batch_size, N)

                print("#"*100)
                print(f"beta: {beta.shape}")

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

                beta_dist = dist.Laplace(torch.zeros(x.shape[1]), torch.ones(x.shape[1]) * tau).to_event(1)
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


def make_lm_program_Laplace_ig_intercept_batched(
        tau: float = 1.0,
        a: float = 5.0,
        b: float = 2.0,
        tau_beta0: float = 3.0
    ) -> 'LM_abstract.pprogram_linear_model_return_dict':
        """
        Make a linear model probabilistic program with an inverse gamma prior on sigma_squared and a Laplace prior on beta.
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
                
                dist_beta0 = dist.Normal(0, tau_beta0**2)


                with pyro.plate("batch", batch_size, dim=-1):   
                        
                        sigma_squared = pyro.sample("sigma_squared", sigma_squared_dist).squeeze() # Shape: (batch_size,)

                        #print(beta_cov.shape)
                        beta_dist = dist.Laplace(torch.zeros(P, 1), torch.ones(P, 1) * tau)  # Shape: (batch_size, P)
                        #print(beta_dist.batch_shape, beta_dist.event_shape)
                        beta = pyro.sample("beta", beta_dist).T  # Shape: (batch_size, P)
                
                        # Compute mean using matrix multiplication
                        mean = torch.matmul(x, beta.unsqueeze(-1)).squeeze(-1)  # Shape: (batch_size, N)


                        with pyro.plate("data", N):
                                noise = pyro.sample("noise", dist.Normal(0, sigma_squared))  # Shape: (batch_size, N)
                        
                        beta0 = pyro.sample("beta0", dist_beta0)
                        beta0 = beta0.unsqueeze(-1)  # Shape: (batch_size, 1)
                        
                        noise = noise.permute(1, 0)  # Shape: (N, batch_size)
                        y = mean + noise  # Shape: (batch_size, N)
                        y = y + beta0 # Shape: (batch_size, N)


                print("#"*100)
                print(f"beta: {beta.shape}")

                sigma_squared = sigma_squared.unsqueeze(-1)
                beta = torch.cat([beta0, beta], dim=-1)

                return {
                                "x": x,
                                "y": y,
                                "sigma_squared": sigma_squared,
                                "beta": beta,
                                'beta0': beta0
                        }

        return multivariate_lm_return_dict

def make_lm_program_Laplace_ig_intercept(
        tau: float = 1.0,
        a: float = 5.0,
        b: float = 2.0,
        tau_beta0: float = 3.0
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

                sigma_squared = pyro.sample("sigma_squared", sigma_squared_dist).squeeze()

                beta_dist = dist.Laplace(torch.zeros(x.shape[1]), torch.ones(x.shape[1]) * tau).to_event(1)
                dist_beta0 = dist.Normal(0, tau_beta0**2)
                beta = pyro.sample("beta", beta_dist)
                beta_0 = pyro.sample("beta0", dist_beta0)

                # Compute mean using matrix multiplication
                mean = torch.matmul(x, beta)

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



def make_lm_program_LaplaceLaplace_ig_batched(
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
                
                        # Compute mean using matrix multiplication
                        mean = torch.matmul(x, beta.unsqueeze(-1)).squeeze(-1)  # Shape: (batch_size, N)


                        with pyro.plate("data", N):
                                noise = pyro.sample("noise", dist.Laplace(0, sigma_squared))  # Shape: (batch_size, N)
                        
                        noise = noise.permute(1, 0)  # Shape: (N, batch_size)
                        print(f"mean shape: {mean.shape}, noise shape: {noise.shape}")
                        y = mean + noise  # Shape: (batch_size, N)
                    
                
                sigma_squared = sigma_squared.unsqueeze(-1)
                


                return {
                                "x": x,
                                "y": y,
                                "sigma_squared": sigma_squared,
                                "beta": beta
                        }

        return multivariate_lm_return_dict

def make_lm_program_LaplaceLaplace_ig(
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

                beta_dist = dist.Laplace(torch.zeros(x.shape[1]), torch.ones(x.shape[1]) * tau).to_event(1)
                beta = pyro.sample("beta", beta_dist)

                # Compute mean using matrix multiplication
                mean = torch.matmul(x, beta)

                with pyro.plate("data", len(x)):
                        y = pyro.sample("obs", pyro.distributions.Laplace(mean, sigma_squared), obs=y)


                return {
                        "x": x,
                        "y": y,
                        "sigma_squared": sigma_squared,
                        "beta": beta
                }

        return multivariate_lm_return_dict



def make_lm_program_Horseshoe_ig_batched(
        gammma_lambda: float = 1.0,
        gamma_tausquared: float = 1.0,
        a: float = 5.0,
        b: float = 2.0
    ) -> 'LM_abstract.pprogram_linear_model_return_dict':
        """
        Make a linear model probabilistic program with a horseshoe prior on beta and an inverse gamma prior on sigma_squared.
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
                lambda_dist = dist.HalfCauchy(gammma_lambda) 
                tau_squared_dist = dist.HalfCauchy(gamma_tausquared)

                with pyro.plate("batch", batch_size, dim=-1):   
                        
                        sigma_squared = pyro.sample("sigma_squared", sigma_squared_dist).squeeze() # Shape: (batch_size,)

                        tau_sq = pyro.sample("tau_sq", tau_squared_dist).squeeze()

                        with pyro.plate("beta_plate", P):
                                lambda_ = pyro.sample("lambda", lambda_dist)
                                beta = pyro.sample("beta", dist.Normal(0, (lambda_**2)* (tau_sq**2)))
                        #print(beta_dist.batch_shape, beta_dist.event_shape)
                       
                        # Compute mean using matrix multiplication
                        
                        beta = beta.permute(1, 0).unsqueeze(1)
                        beta = beta.transpose(-2, -1)
                        mean = torch.matmul(x, beta).squeeze()

                        beta = beta.squeeze()

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


def make_lm_program_Horseshoe_ig(
        gammma_lambda: float = 1.0,
        gamma_tausquared: float = 1.0,
        a: float = 5.0,
        b: float = 2.0
    ) -> 'LM_abstract.pprogram_linear_model_return_dict':
        """
        Make a linear model probabilistic program with a horseshoe prior on beta and an inverse gamma prior on sigma_squared.
        Args: 
                tau: float: the standard deviation of the prior on beta
                a: float: the shape parameter of the gamma prior on sigma_squared
                b: float: the rate parameter of the gamma prior on sigma_squared
        Returns:
                LM_abstract.pprogram_linear_model_return_dict: a linear model probabilistic program
        """
        def multivariate_lm_return_dict(x: torch.Tensor, y: torch.Tensor = None) -> dict:
                N, P = x.shape

                # Define distributions for the global parameters
                sigma_squared_dist = dist.InverseGamma(a, b)
                lambda_dist = dist.HalfCauchy(gammma_lambda) 
                tau_squared_dist = dist.HalfCauchy(gamma_tausquared)

                        
                sigma_squared = pyro.sample("sigma_squared", sigma_squared_dist).squeeze() # Shape: (batch_size,)

                tau_sq = pyro.sample("tau_sq", tau_squared_dist).squeeze()

                with pyro.plate("beta_plate", P):
                        lambda_ = pyro.sample("lambda", lambda_dist)
                        beta = pyro.sample("beta", dist.Normal(0, (lambda_**2) * tau_sq))
                        #print(beta_dist.batch_shape, beta_dist.event_shape)
                       
                        # Compute mean using matrix multiplication
                        
            
                mean = torch.matmul(x, beta).squeeze(-1)  # Shape: (batch_size, N)

                with pyro.plate("data", N):
                        y = pyro.sample("obs", pyro.distributions.Normal(mean, sigma_squared), obs=y)

                sigma_squared = sigma_squared.unsqueeze(-1)

                return {
                                "x": x,
                                "y": y,
                                "sigma_squared": sigma_squared,
                                "beta": beta
                        }

        return multivariate_lm_return_dict