from PFNExperiments.LinearRegression.Models.ModelPosterior import ModelPosterior
import torch 

class ModelToPosterior:
    """
    Class that takes in a torch.nn.Module and a posterior and returns the posterior given an input of batches to the model 
    """

    def __init__(self,
                 model: torch.nn.Module,
                 posterior_model: ModelPosterior) -> None:
        """
        Args:
            model: torch.nn.Module: the model
            posterior_model: ModelPosterior: the posterior model
        """
        self.model = model
        self.posterior_model = posterior_model


    def input_to_posterior(self, x: torch.Tensor) -> torch.distributions.distribution.Distribution:
        """
        A method that takes in the input and returns the posterior distribution
        Args:
            x: torch.Tensor: the input
        Returns:
            torch.distributions.distribution.Distribution: the posterior distribution
        """
        pred = self.model(x)
        posterior = self.posterior_model.pred2posterior(pred)
        return posterior
    