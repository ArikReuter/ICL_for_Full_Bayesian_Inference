import torch 
from tqdm import tqdm

from torchdiffeq import odeint
from torchdiffeq import odeint_adjoint

from PFNExperiments.LinearRegression.Evaluation.CompareComparisonModels import PosteriorComparisonModel
from PFNExperiments.Training.FlowMatching.DDPMLossDiffusionVP import DDPMLossDiffusionVP

class ModelToPosteriorCNF(PosteriorComparisonModel):
    """
    A class that allows to generate samples from a model that is trained with flow matching and outputs a vector field at each time step t
    The model is required to take in the forward pass a triple of inputs (z, x, t) and return a vector field
    """

    def sample_from_base_distribution_standard_normal(shape: torch.Size) -> torch.Tensor:
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
                 sample_shape: torch.Size,
                 sample_name: str = "beta",
                 n_samples: int = 1000,
                 batch_size: int = 256,
                 base_dist_sample_function: callable = sample_from_base_distribution_standard_normal,
                 solver: str = 'dopri5',
                 atol: float = 1e-7,
                 rtol: float = 1e-7,
                 device: str = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
                 target_device: str = torch.device("cpu"),
                 solve_adjoint: bool = False,
                 epsilon_for_t: float = 0.0, 
                 score_matching: bool = False,
                 score_matching_loss: DDPMLossDiffusionVP = None
                 ) -> None:
        """
        Args:
            model: torch.nn.Module: the model that is trained with flow matching and takes in the forward pass a triple of inputs (z, x, t) and returns a vector field
            sample_shape: torch.Size: the shape of the samples that are generated
            sample_name: str: the name of the sample
            n_samples: int: the number of samples to generate
            batch_size: int: the batch size for the ODE solver
            solver: str: the solver to use for the ODE solver
            atol: float: the absolute tolerance for the ODE solver
            rtol: float: the relative tolerance for the ODE solver
            adjoint: bool: whether to use the adjoint method for the ODE solver
            device: str: the device to use for the computation
            tarfet_device: str: the device to use for the final output samples
            epsilon_for_t: float: the epsilon for the time step that scales the time to be in [0, 1 - epsilon_for_t]
            score_matching: bool: whether to use score matching for the loss
            score_matching_loss: DDPMLossDiffusionVP: the loss function for score matching
        """
        self.model = model.to(device)
        self.sample_shape = sample_shape
        self.sample_name = sample_name
        self.base_dist_sample_function = base_dist_sample_function
        self.solver = solver
        self.atol = atol
        self.rtol = rtol
        self.device = device
        self.target_device = target_device

        self.n_samples = n_samples
        self.batch_size = batch_size
        self.solve_adjoint = solve_adjoint
        self.epsilon_for_t = epsilon_for_t
        self.score_matching = score_matching
        self.score_matching_loss = score_matching_loss

        if self.score_matching is True:
            assert self.score_matching_loss is not None, "Score matching loss is not provided"

    def generate_vector_field_function_cond_x(self, x: torch.tensor) -> torch.nn.Module:
        """
        Generate the vector field function that takes in the input x and returns the vector field at time t
        Args:
            x: torch.tensor: the input to the model that conditions the distribution
        """

        if not self.score_matching:

            class VectorFieldFunction(torch.nn.Module):
                def __init__(self, model: torch.nn.Module, x: torch.tensor):
                    super(VectorFieldFunction, self).__init__()
                    self.model = model
                    self.x = x

                def forward(self, t, z):
                    if len(t.shape) == 0:
                        t = t.unsqueeze(0)
                        t = t.repeat(z.shape[0], 1)

                    return self.model(z, self.x, t)
                
            return VectorFieldFunction(self.model, x)
        
        else:

            model2vf = self.score_matching_loss.model_prediction_to_vector_field

            class VectorFieldFunction(torch.nn.Module):
                def __init__(self, model: torch.nn.Module, x: torch.tensor):
                    super(VectorFieldFunction, self).__init__()
                    self.model = model
                    self.x = x

                def forward(self, t, z):
                    if len(t.shape) == 0:
                        t = t.unsqueeze(0)
                        t = t.repeat(z.shape[0], 1)

                    vf = model2vf(
                        model=self.model,
                        z=z,
                        x=self.x,
                        t=t
                    )

                    return vf
                
            return VectorFieldFunction(self.model, x)


    def sample_posterior_x_batch(self, x: torch.tensor, n_samples: int) -> torch.Tensor:
        """
        Return samples from the posterior by solving the ODE with the learned vector field
        this processes a single batch 
        Args:
            x: torch.tensor: the input to the model that conditions the distribution
            n_samples: int: the number of samples to generate
        """


        # duplicate the input x to match the number of samples
        x = x.unsqueeze(0).repeat(n_samples, 1, 1)

        z_0 = self.base_dist_sample_function((n_samples, *self.sample_shape))

        z_0 = z_0.to(self.device)
        x = x.to(self.device)

        timepoints = torch.tensor([self.epsilon_for_t, 1.0 - self.epsilon_for_t]).to(self.device)

        vector_field_function = self.generate_vector_field_function_cond_x(x)
        vector_field_function = vector_field_function.to(x.device)

        if not self.solve_adjoint:
            samples = odeint(vector_field_function, z_0, timepoints, atol=self.atol, rtol=self.rtol, method=self.solver)

        else:
            samples = odeint_adjoint(vector_field_function, z_0, timepoints, atol=self.atol, rtol=self.rtol, method=self.solver)


        samples = samples.to(self.target_device)

        return samples[-1]

    def sample_posterior_x(self, X: torch.Tensor, n_samples: int) -> torch.Tensor:
        """
        Return samples from the posterior by solving the ODE with the learned vector field
        Args:
            X: torch.Tensor: the input to the model that conditions the distribution
            n_samples: int: the number of samples to generate
        """

        n_batches = int(n_samples / self.batch_size)
        res = []

        for i in tqdm(list(range(n_batches))):
            x = X[i * self.batch_size: (i + 1) * self.batch_size]
            res.append(self.sample_posterior_x_batch(x, self.batch_size))

        if n_samples % self.batch_size != 0:
            x = X[n_batches * self.batch_size: n_samples]
            res.append(self.sample_posterior_x_batch(x, n_samples % self.batch_size))

        return torch.cat(res, dim=0)

    
    def sample_posterior(self,  
                X: torch.Tensor,
                y: torch.Tensor = None) -> torch.Tensor:
        """
        A method that samples from the posterior distribution
        Args:
            X: torch.Tensor: the covariates
            y: torch.Tensor: the response variable
        Returns:
            torch.Tensor
        """
        if y is not None:
            if len(y.shape) == 1:
                y = y.unsqueeze(-1)


            X_y = torch.cat([X, y], dim = -1) # concatenate the x and y values to one data tensor
        else:
            X_y = X
            
        samples = self.sample_posterior_x(X_y, self.n_samples)

        res = {
            self.sample_name: samples,
            "X": X,
            "y": y
        }

        return res