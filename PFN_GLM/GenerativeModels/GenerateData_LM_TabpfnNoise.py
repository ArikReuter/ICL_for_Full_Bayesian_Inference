import pyro.distributions
import torch 
import pyro 
import pyro.distributions as dist

#from PFNLinearRegression.GenerativeModels import LM_abstract


def normalize(T: torch.Tensor) -> torch.Tensor:
        """
        Normalize a tensor elementwise.
        Args:
                T: torch.Tensor: the tensor to normalize
        Returns:
                torch.Tensor: the normalized tensor
        """

        T = T - T.mean()
        T = T / T.std()

        return T

def normalize_per_dataset_per_feature(T: torch.Tensor, eps:float = 1e-6) -> torch.Tensor:
        """
        Normalize a tensor per dataset and per feature.
        Args:
                T: torch.Tensor: the tensor to normalize of shape (N_datasets, N_samples, N_features)
                eps: float: a small number to avoid division by zero
        Returns:
                torch.Tensor: the normalized tensor
        """

        T = T - T.mean(dim=1, keepdim=True)
        T = T / (T.std(dim=1, keepdim=True) + eps)

        return T



class make_lm_program_ig_intercept_tabpfn_noise_batched:
        
        def __init__(self,
                     data_stored: torch.Tensor):
                """
                Make a linear model probabilistic program with a gamma prior on sigma_squared and a gamma prior on beta_var.
                The noise is sampled from the tabpfn prior.
                Args:
                        data_stored: torch.tensor: the tensor to load the covariates from. Has shape (N_datasets, N_samples, N_features)
                """
                data_stored = normalize_per_dataset_per_feature(data_stored)
                self.data_stored = data_stored

        def __call__(
                self,
                tau: float = 1.0,
                a: float = 5.0,
                b: float = 2.0,
                tau_beta0: float = 3.0,
        ) -> 'LM_abstract.pprogram_linear_model_return_dict':
                """
                Make a linear model probabilistic program with a gamma prior on sigma_squared and a gamma prior on beta_var.
                The noise is sampled from the tabpfn prior.
                Args:   
                        Data_stored: torch.tensor: the tensor to load the covariates from. Has shape (N_datasets, N_samples, N_features)
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


                                dataset_indices_noise = torch.randint(0, self.data_stored.shape[0], (batch_size,)) # Shape: (batch_size,) # select one dataset for each element in the batch

                                feature_indices_noise = torch.randint(0, self.data_stored.shape[2], (batch_size, )) # Shape: (batch_size,) # select one feature for each element in the batch

                                #sample_indices_noise = torch.randint(0, self.data_stored.shape[1], (batch_size, N)) # Shape: (batch_size, N) # select N samples for each element in the batch
                                
                                noise = self.data_stored[dataset_indices_noise, :, feature_indices_noise] # Shape: (batch_size, N)

                        
                                # randomly arange the noise
                                noise = noise[:, torch.randperm(noise.size(1))]
                                noise = noise[:, :N] # Shape: (batch_size, N)

                                noise = sigma_squared.unsqueeze(-1) * noise

                                beta0 = pyro.sample("beta0", dist_beta0)
                                beta0 = beta0.unsqueeze(-1)  # Shape: (batch_size, 1)
                                
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