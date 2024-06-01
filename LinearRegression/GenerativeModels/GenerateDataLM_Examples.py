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

def make_lm_program_plain(
        beta_var: float = 1.0,
        sigma_squared: float = 0.1
        ) -> LM_abstract.pprogram_linear_model_return_dict:
    """
    Make a very simple linear model probabilistic program.
        Args:
                beta_var: float: the variance of the parameters of the linear model
                sigma_squared: float: the variance of the response variable
        Returns:
                LM_abstract.pprogram_linear_model_return_dict: a linear model probabilistic program
    """
    sigma_squared = torch.tensor(sigma_squared)
    def multivariate_lm_return_dict(x: torch.tensor, y: torch.tensor = None) -> dict:

        beta_cov = torch.eye(x.shape[1]) * beta_var # the covariance matrix of the parameters of the linear model
        beta = pyro.sample("beta", pyro.distributions.MultivariateNormal(torch.zeros(x.shape[1]), beta_cov)) # the parameters of the linear model

        mean = torch.matmul(x, beta)

        with pyro.plate("data", x.shape[0]):
            y = pyro.sample("y", pyro.distributions.Normal(mean, sigma_squared))

        return {
                "x": x,
                "y": y,
                "beta": beta
                }
    
    return multivariate_lm_return_dict

def make_lm_program_gamma_gamma(
        a0:float = 5.0,
        b0: float = 2.0,
        a1: float = 5.0,
        b1: float = 2.0
        ) -> LM_abstract.pprogram_linear_model_return_dict:
        """
        Make a linear model probabilistic program with a gamma prior on sigma_squared and a gamma prior on beta_var.
        Args: 
                a0: float: the shape parameter of the gamma prior on beta_var
                bo: float: the rate parameter of the gamma prior on beta_var
                a1: float: the shape parameter of the gamma prior on sigma_squared
                b1: float: the rate parameter of the gamma prior on sigma_squared
        Returns:
                LM_abstract.pprogram_linear_model_return_dict: a linear model probabilistic program
        """
        def multivariate_lm_return_dict(x: torch.tensor, y: torch.tensor = None) -> dict:
            
            beta_dist = pyro.distributions.Gamma(a0, b0)
            beta_var = pyro.sample("beta_var", beta_dist)
        
            sigma_squared_dist = pyro.distributions.Gamma(a1, b1)
            sigma_squared = pyro.sample("sigma_squared", sigma_squared_dist)

            beta_cov = torch.eye(x.shape[1]) * beta_var # the covariance matrix of the parameters of the linear model
            beta = pyro.sample("beta", pyro.distributions.MultivariateNormal(torch.zeros(x.shape[1]), beta_cov)) # the parameters of the linear model

            mean = torch.matmul(x, beta)

            with pyro.plate("data", x.shape[0]):
                y = pyro.sample("obs", pyro.distributions.Normal(mean, sigma_squared), obs=y)

            return {
                        "x": x,
                        "y": y,
                        "sigma_squared": sigma_squared,
                        "beta_var": beta_var,
                        "beta": beta
                }
        
        return multivariate_lm_return_dict





import torch
import pyro
import pyro.distributions as dist

def make_lm_program_gamma_gamma_batched(
        a0: float = 5.0,
        b0: float = 2.0,
        a1: float = 5.0,
        b1: float = 2.0
    ) -> 'LM_abstract.pprogram_linear_model_return_dict':
        """
        Make a linear model probabilistic program with a gamma prior on sigma_squared and a gamma prior on beta_var.
        Args: 
                a0: float: the shape parameter of the gamma prior on beta_var
                b0: float: the rate parameter of the gamma prior on beta_var
                a1: float: the shape parameter of the gamma prior on sigma_squared
                b1: float: the rate parameter of the gamma prior on sigma_squared
        Returns:
                LM_abstract.pprogram_linear_model_return_dict: a linear model probabilistic program
        """
        def multivariate_lm_return_dict(x: torch.Tensor, y: torch.Tensor = None) -> dict:
                if x.dim() == 2:
                        x = x.unsqueeze(0)  # Ensure x is 3D (batch_size, N, P)
                batch_size, N, P = x.shape

                # Define distributions for the global parameters
                beta_dist = dist.Gamma(a0, b0)
                sigma_squared_dist = dist.Gamma(a1, b1)

                with pyro.plate("batch", batch_size, dim=-1):
                        # Sample global parameters per batch
                        beta_var = pyro.sample("beta_var", beta_dist).squeeze() # Shape: (batch_size,)
                        
                        sigma_squared = pyro.sample("sigma_squared", sigma_squared_dist).squeeze() # Shape: (batch_size,)
                        # Create beta covariance matrix based on the number of covariates P
                        beta_cov = torch.eye(P) * beta_var.unsqueeze(-1).unsqueeze(-1)  # Expand beta_var to shape (batch_size, P, P)
                        
                        #print(beta_cov.shape)
                        beta_dist = dist.MultivariateNormal(torch.zeros(P), beta_cov)
                        #print(beta_dist.batch_shape, beta_dist.event_shape)
                        beta = pyro.sample("beta", beta_dist)  # Shape: (batch_size, P)

                
                        # Compute mean using matrix multiplication
                        mean = torch.matmul(x, beta.unsqueeze(-1)).squeeze(-1)  # Shape: (batch_size, N)


                        with pyro.plate("data", N):
                                noise = pyro.sample("noise", dist.Normal(0, sigma_squared))  # Shape: (batch_size, N)
                        
                        noise = noise.permute(1, 0)  # Shape: (N, batch_size)
                        y = mean + noise  # Shape: (batch_size, N)



                return {
                                "x": x,
                                "y": y,
                                "sigma_squared": sigma_squared,
                                "beta_var": beta_var,
                                "beta": beta
                        }

        return multivariate_lm_return_dict


def make_make_lm_program_gamma_gamma_batched_quantized(
            quantizer: Quantizer
        ):
        def make_lm_program_gamma_gamma_batched_quantized(
                a0: float = 5.0,
                b0: float = 2.0,
                a1: float = 5.0,
                b1: float = 2.0,
                log_n_bins: float = 1.0
        ) -> LM_abstract.pprogram_linear_model_return_dict:
                """
                Make a linear model probabilistic program with a gamma prior on sigma_squared and a gamma prior on beta_var.
                Args: 
                        a0: float: the shape parameter of the gamma prior on beta_var
                        b0: float: the rate parameter of the gamma prior on beta_var
                        a1: float: the shape parameter of the gamma prior on sigma_squared
                        b1: float: the rate parameter of the gamma prior on sigma_squared
                        log_n_bins: float: the log of the number of bins to quantize the parameters of the linear model

                Returns:
                        LM_abstract.pprogram_linear_model_return_dict: a linear model probabilistic program
                """
                log_n_bins = int(log_n_bins)
                def multivariate_lm_return_dict(x: torch.Tensor, y: torch.Tensor = None) -> dict:
                        if x.dim() == 2:
                                x = x.unsqueeze(0)  # Ensure x is 3D (batch_size, N, P)
                        batch_size, N, P = x.shape

                        # Define distributions for the global parameters
                        beta_dist = dist.Gamma(a0, b0)
                        sigma_squared_dist = dist.Gamma(a1, b1)

                        
                        with pyro.plate("batch", batch_size, dim=-1):
                                # Sample global parameters per batch
                                beta_var = pyro.sample("beta_var", beta_dist).squeeze() # Shape: (batch_size,)
                                
                                sigma_squared = pyro.sample("sigma_squared", sigma_squared_dist).squeeze() # Shape: (batch_size,)
                                # Create beta covariance matrix based on the number of covariates P
                                beta_cov = torch.eye(P) * beta_var.unsqueeze(-1).unsqueeze(-1)  # Expand beta_var to shape (batch_size, P, P)
                                
                                #print(beta_cov.shape)
                                beta_dist = dist.MultivariateNormal(torch.zeros(P), beta_cov)
                                #print(beta_dist.batch_shape, beta_dist.event_shape)
                                beta = pyro.sample("beta", beta_dist)  # Shape: (batch_size, P)
                                
                                
                                beta_quant = quantizer.quantize(beta, n_buckets=2**log_n_bins)
                        
                                # Compute mean using matrix multiplication
                                mean = torch.matmul(x, beta_quant.unsqueeze(-1)).squeeze(-1)  # Shape: (batch_size, N)


                                with pyro.plate("data", N):
                                        noise = pyro.sample("noise", dist.Normal(0, sigma_squared))  # Shape: (batch_size, N)
                                
                                noise = noise.permute(1, 0)  # Shape: (N, batch_size)
                                y = mean + noise  # Shape: (batch_size, N)
                                
                                

                        return {
                                        "x": x,
                                        "y": y,
                                        "sigma_squared": sigma_squared,
                                        "beta_var": beta_var,
                                        "beta": beta_quant
                                }

                return multivariate_lm_return_dict
        
        return make_lm_program_gamma_gamma_batched_quantized





def make_lm_program_gamma_gamma_reduced_dimensionality(
        a0:float = 5.0,
        b0: float = 2.0,
        a1: float = 5.0,
        b1: float = 2.0,
        dimensionality_factor: float = 1.0,
        ) -> LM_abstract.pprogram_linear_model_return_dict:
        """
        Make a linear model probabilistic program with a gamma prior on sigma_squared and a gamma prior on beta_var.
        The dimensionality of the parameters of the linear model is the fraction of covariates that have an effec that is not zero.
        Args:
                a0: float: the shape parameter of the gamma prior on beta_var
                bo: float: the rate parameter of the gamma prior on beta_var
                a1: float: the shape parameter of the gamma prior on sigma_squared
                b1: float: the rate parameter of the gamma prior on sigma_squared
                dimensionality_factor: float: the fraction of covariates that have an effect that is not zero
        Returns:
                LM_abstract.pprogram_linear_model_return_dict: a linear model probabilistic program
        """

        def multivariate_lm_return_dict(x: torch.tensor, y: torch.tensor = None) -> dict:
            
            beta_dist = pyro.distributions.Gamma(a0, b0)
            beta_var = pyro.sample("beta_var", beta_dist)
        
            sigma_squared_dist = pyro.distributions.Gamma(a1, b1)
            sigma_squared = pyro.sample("sigma_squared", sigma_squared_dist)

            

            beta_cov = torch.eye(x.shape[1]) * beta_var # the covariance matrix of the parameters of the linear model
            beta = pyro.sample("beta", pyro.distributions.MultivariateNormal(torch.zeros(x.shape[1]), beta_cov)) # the parameters of the linear model

            index_till_zero = int(x.shape[1] * dimensionality_factor)
            beta[index_till_zero:] = 0

            mean = torch.matmul(x, beta)

            with pyro.plate("data", x.shape[0]):
                y = pyro.sample("obs", pyro.distributions.Normal(mean, sigma_squared), obs=y)

            return {
                        "x": x,
                        "y": y,
                        "sigma_squared": sigma_squared,
                        "beta_var": beta_var,
                        "beta": beta
                }
        
        return multivariate_lm_return_dict


def make_lm_program_gamma_gamma_augmented(
        a0:float = 5.0,
        b0: float = 2.0,
        a1: float = 5.0,
        b1: float = 2.0
        ) -> LM_abstract.pprogram_linear_model_return_dict:
        """
        Make a linear model probabilistic program with a gamma prior on sigma_squared and a gamma prior on beta_var.
        Here, specify the prior on tau, the precision of the parameters of the linear model. and do some additional changes. 
        Args: 
                a0: float: the shape parameter of the gamma prior on beta_var
                bo: float: the rate parameter of the gamma prior on beta_var
                a1: float: the shape parameter of the gamma prior on sigma_squared
                b1: float: the rate parameter of the gamma prior on sigma_squared
        Returns:
                LM_abstract.pprogram_linear_model_return_dict: a linear model probabilistic program
        """
        a0 = torch.tensor(a0)
        b0 = torch.tensor(b0)
        a1 = torch.tensor(a1)
        b1 = torch.tensor(b1)
        def multivariate_lm_return_dict(x: torch.tensor, y: torch.tensor = None) -> dict:
                sigma_squared = (pyro.sample("sigma_squared", pyro.distributions.Gamma(a0, b0)) + 0.1)/10000
                
                tau = (pyro.sample("tau", pyro.distributions.Gamma(a1,b1)) + 0.1)/10

                beta_mean = torch.zeros(x.shape[1])
                beta_cov = (1/tau) * torch.eye(x.shape[1])

                beta = pyro.sample("beta", pyro.distributions.MultivariateNormal(beta_mean, beta_cov))

                mean = torch.matmul(x, beta)

                with pyro.plate("data", len(x)):
                        y = pyro.sample("obs", pyro.distributions.Normal(mean, sigma_squared), obs=y)

                return {
                        "x": x,
                        "y": y,
                        "sigma_squared": sigma_squared,
                        "tau": tau,
                        "beta": beta
                }

        return multivariate_lm_return_dict


def make_lm_program_ig_batched(
        tau: float = 1.0,
        a: float = 5.0,
        b: float = 2.0
    ) -> 'LM_abstract.pprogram_linear_model_return_dict':
        """
        Make a linear model probabilistic program with a gamma prior on sigma_squared and a gamma prior on beta_var.
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


                beta_cov = torch.eye(P) * (tau ** 2)  # the covariance matrix of the parameters of the linear model

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
                        
                        noise = noise.permute(1, 0)  # Shape: (N, batch_size)
                        y = mean + noise  # Shape: (batch_size, N)



                return {
                                "x": x,
                                "y": y,
                                "sigma_squared": sigma_squared,
                                "beta": beta
                        }

        return multivariate_lm_return_dict

def make_lm_program_ig(
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

                beta_cov = torch.eye(x.shape[1]) * (tau ** 2)  # the covariance matrix of the parameters of the linear model

                sigma_squared = pyro.sample("sigma_squared", sigma_squared_dist).squeeze()

                beta_dist = dist.MultivariateNormal(torch.zeros(x.shape[1]), beta_cov)
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


def make_lm_program_spike_and_slap_batched(
              pi: float = 0.5, 
              sigma_squared_outer: float = 0.1,
              beta_var: float = 1.0
              ) -> 'LM_abstract.pprogram_linear_model_return_dict':
        """
        Make a linear model probabilistic program with a spike and slab prior on beta.
        Args:
                pi: float: the probability of the spike
                sigma_squared: float: the variance of the response variable
                beta_var: float: the variance of the parameters of the linear model
        Returns:
                LM_abstract.pprogram_linear_model_return_dict: a linear model probabilistic program
        """

        def multivariate_lm_return_dict(x: torch.Tensor, y: torch.Tensor = None) -> dict:
                if x.dim() == 2:
                        x = x.unsqueeze(0)  # Ensure x is 3D (batch_size, N, P)
                batch_size, N, P = x.shape

                # Define distributions for the global parameters

                beta_cov = torch.eye(P) * beta_var  # the covariance matrix of the parameters of the linear model
                beta_dist = pyro.distributions.MultivariateNormal(torch.zeros(P), beta_cov)
                sigma_squared= torch.tensor(sigma_squared_outer)

                include_beta = pyro.sample("include_beta", pyro.distributions.Bernoulli(pi).expand([batch_size, P]).to_event(1))                        

                with pyro.plate("batch", batch_size, dim=-1):
                        # Sample global parameters per batch
                        beta = pyro.sample("beta", beta_dist)  # Shape: (batch_size, P)

                        
                        beta = beta * include_beta # Shape: (batch_size, P)

                        # Compute mean using matrix multiplication
                        mean = torch.matmul(x, beta.unsqueeze(-1)).squeeze(-1)  # Shape: (batch_size, N)

                        with pyro.plate("data", N):
                                noise = pyro.sample("noise", dist.Normal(0, sigma_squared))  # Shape: (batch_size, N)
                        
                        noise = noise.permute(1, 0)  # Shape: (N, batch_size)
                        y = mean + noise  # Shape: (batch_size, N)

                return {
                                "x": x,
                                "y": y,
                                "sigma_squared": sigma_squared,
                                "beta": beta
                        }
        
        return multivariate_lm_return_dict


def make_lm_program_spike_and_slap(
                  pi: float = 0.5, 
                  sigma_squared_outer: float = 0.1,
                  beta_var: float = 1.0
                  ) -> 'LM_abstract.pprogram_linear_model_return_dict':
        """
        Make a linear model probabilistic program with a spike and slab prior on beta.
        Args:
            pi: float: the probability of the spike
                    sigma_squared: float: the variance of the response variable
                    beta_var: float: the variance of the parameters of the linear model
        Returns:
                    LM_abstract.pprogram_linear_model_return_dict: a linear model probabilistic program
        """
        def multivariate_lm_return_dict(x: torch.Tensor, y: torch.Tensor = None) -> dict:
                # Define distributions for the global parameters
                beta_cov = torch.eye(x.shape[1]) * beta_var
                beta_dist = pyro.distributions.MultivariateNormal(torch.zeros(x.shape[1]), beta_cov)
                sigma_squared= torch.tensor(sigma_squared_outer)

                include_beta = pyro.sample("include_beta", pyro.distributions.Bernoulli(pi).expand([x.shape[1]]).to_event(1))

                beta = pyro.sample("beta", beta_dist)  # the parameters of the linear model
                beta = beta * include_beta

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

