import torch 
from tqdm import tqdm

from torchdiffeq import odeint
from torchdiffeq import odeint_adjoint

from PFNExperiments.LinearRegression.Evaluation.CompareComparisonModels import PosteriorComparisonModel

class ModelToPosteriorCNF_LearnedBaseDist(PosteriorComparisonModel):
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
                 sample_only_base_dist: bool = False
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
            solve_adjoint: bool: whether to solve the ODE with the adjoint method
            sample_only_base_dist: bool: whether to only sample from the base distribution and not use the flow matching model
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

        self.sample_only_base_dist = sample_only_base_dist

    def __repr__(self) -> str:
        return "ModelToPosteriorCNF_LearnedBaseDist"

    def generate_vector_field_function_cond_x(self, x: torch.tensor) -> torch.nn.Module:
        """
        Generate the vector field function that takes in the input x and returns the vector field at time t
        Args:
            x: torch.tensor: the input to the model that conditions the distribution
        """

        class VectorFieldFunction(torch.nn.Module):
            def __init__(self, model: torch.nn.Module, encoder_representation: torch.tensor):
                super(VectorFieldFunction, self).__init__()
                self.model = model
                self.encoder_representation = encoder_representation

            def forward(self, t, z):
                if len(t.shape) == 0:
                    t = t.unsqueeze(0)
                    t = t.repeat(z.shape[0], 1)

                vector_field_prediction = self.model.forward_decoder(  # compute the vector field prediction by the model
                    z = z,
                    x_encoder = self.encoder_representation,
                    condition_time= t
                )

                return vector_field_prediction
            
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

        #z_0 = self.base_dist_sample_function((n_samples, *self.sample_shape))

        #z_0 = z_0.to(self.device)
        x = x.to(self.device)

        timepoints = torch.tensor([0., 1.]).to(self.device)

        encoder_prediction, encoder_representation = self.model.forward_encoder(x)  # get the final output of the encoder and the intermediate transformer representation
        base_distribution_samples = self.model.get_base_distribution_samples(encoder_prediction)  # sample from the base distribution

        vector_field_function = self.generate_vector_field_function_cond_x(encoder_representation)
        vector_field_function = vector_field_function.to(x.device)

        if not self.solve_adjoint:
            samples = odeint(vector_field_function, base_distribution_samples, timepoints, atol=self.atol, rtol=self.rtol, method=self.solver)

        else:
            samples = odeint_adjoint(vector_field_function, base_distribution_samples, timepoints, atol=self.atol, rtol=self.rtol, method=self.solver)


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


        if self.sample_only_base_dist:
            x = X_y.unsqueeze(0).repeat(self.n_samples, 1, 1)

            #z_0 = self.base_dist_sample_function((n_samples, *self.sample_shape))

            #z_0 = z_0.to(self.device)
            x = x.to(self.device)

            timepoints = torch.tensor([0., 1.]).to(self.device)

            encoder_prediction, encoder_representation = self.model.forward_encoder(x)  # get the final output of the encoder and the intermediate transformer representation
            base_distribution_samples = self.model.get_base_distribution_samples(encoder_prediction)  # sample from the base distribution

            res = {
                self.sample_name: base_distribution_samples,
                "X": X,
                "y": y
            }
            return res
        
        else: 

            samples = self.sample_posterior_x(X_y, self.n_samples)

            res = {
                self.sample_name: samples,
                "X": X,
                "y": y
            }

            return res