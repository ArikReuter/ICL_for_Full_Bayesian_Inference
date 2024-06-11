import torch 
import torchdiffeq

from torchdiffeq import odeint

from PFNExperiments.LinearRegression.Evaluation.CompareComparisonModels import PosteriorComparisonModel

class ModelToPosteriorCNF(PosteriorComparisonModel):
    """
    A class that allows to generate samples from a model that is trained with flow matching and outputs a vector field at each time step t
    The model is required to take in the forward pass a triple of inputs (z, x, t) and return a vector field
    """

    def sample_from_base_distribution_standard_normal(self, shape: torch.Size) -> torch.Tensor:
        """
        Sample from the base distribution (standard normal)
        Args:
            shape: torch.Size: the shape of the sample
        Returns:
            torch.Tensor: the sample
        """
        return torch.randn(shape)


    def __init__(self,
                 model: torch.nn.Module,
                 n_samples: int = 1000,
                 base_dist_sample_function: callable = sample_from_base_distribution_standard_normal,
                 solver: str = 'dopri5',
                 atol: float = 1e-5,
                 rtol: float = 1e-5,
                 adjoint: bool = False,
                 ) -> None:
        """
        Args:
            model: torch.nn.Module: the model that is trained with flow matching and takes in the forward pass a triple of inputs (z, x, t) and returns a vector field
            timesteps: int: the number of timesteps to generate the samples
            solver: str: the solver to use for the ODE solver
            atol: float: the absolute tolerance for the ODE solver
            rtol: float: the relative tolerance for the ODE solver
            adjoint: bool: whether to use the adjoint method for the ODE solver
        """
        self.model = model
        self.base_dist_sample_function = base_dist_sample_function
        self.solver = solver
        self.atol = atol
        self.rtol = rtol
        self.adjoint = adjoint

        self.n_samples = n_samples

    def generate_vector_field_function_cond_x(self, x: torch.tensor) -> torch.nn.Module:
        """
        Generate the vector field function that takes in the input x and returns the vector field at time t
        Args:
            x: torch.tensor: the input to the model that conditions the distribution
        """

        class VectorFieldFunction(torch.nn.Module):
            def __init__(self, model: torch.nn.Module, x: torch.tensor):
                super(VectorFieldFunction, self).__init__()
                self.model = model
                self.x = x

            def forward(self, t, z):
                return self.model(z, self.x, t)
            
        return VectorFieldFunction(self.model, x)

    def sample_posterior_x(self, x: torch.tensor, n_samples: int) -> torch.Tensor:
        """
        Return samples from the posterior by solving the ODE with the learned vector field
        Args:
            x: torch.tensor: the input to the model that conditions the distribution
            n_samples: int: the number of samples to generate
        """

        z_0 = self.base_dist_sample_function((n_samples, x.shape[0]))
        z_0 = z_0.to(x.device)

        vector_field_function = self.generate_vector_field_function_cond_x(x)
        vector_field_function = vector_field_function.to(x.device)
        samples = odeint(vector_field_function, z_0, torch.tensor([0., 1.]), atol=self.atol, rtol=self.rtol, method=self.solver, adjoint=self.adjoint)

        return samples[-1]

    
    def sample_posterior(self,  
                X: torch.Tensor,
                y: torch.Tensor) -> torch.Tensor:
        """
        A method that samples from the posterior distribution
        Args:
            X: torch.Tensor: the covariates
            y: torch.Tensor: the response variable
        Returns:
            torch.Tensor
        """
        if len(y.shape) == 1:
            y = y.unsqueeze(-1)


        X_y = torch.cat([X, y], dim = -1) # concatenate the x and y values to one data tensor

        samples = self.sample_posterior_x(X_y, self.n_samples)

        return samples