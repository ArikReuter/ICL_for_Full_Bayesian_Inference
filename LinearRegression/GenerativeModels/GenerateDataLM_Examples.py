import torch 
import pyro 
try:
        import LM_abstract
except:
      from PFNExperiments.LinearRegression.GenerativeModels import LM_abstract 

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