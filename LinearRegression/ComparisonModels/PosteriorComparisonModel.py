from abc import ABC, abstractmethod
import torch 
from PFNExperiments.LinearRegression.GenerativeModels.LM_abstract import ppgram_linear_model_return_y

class PosteriorComparisonModel(ABC):
    """
    Class that takes in a model and returns the posterior the model provides
    """

    def __init__(self, 
                 pprogram: ppgram_linear_model_return_y)-> None:
        """
        Args:
            pprogram: ppgram_linear_model_return_y: the probabilistic program
        """
        self.pprogram = pprogram


    @abstractmethod
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

        pass