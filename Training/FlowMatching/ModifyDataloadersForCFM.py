import torch 
from torch.utils.data import Dataset
from typing import Callable

class AddTimePointToDataset():
    """
    A class that adds a time point to the data loaded by the dataset
    """

    def sample_uniform_time() -> float:	
        """
        Just sample a uniform time point
        """
        return torch.rand(1)
        
    def __init__(self,
                 sample_time: Callable = sample_uniform_time):
        """
        Args:
            time_distribution: Callable: the function that determines how the time should be sampled for an element in the dataloader
        """
        self.sample_time = sample_time


    def __call__(self, dataset: torch.utils.data.Dataset) -> torch.utils.data.Dataset:
        """
        Add a time point to the dataset
        Args:
            dataset: torch.utils.data.Dataset: the dataset. The dataset yields a dictionary with the data
        Returns:
            torch.utils.data.Dataset: the dataset with the time point
        """
        
        def new_getitem(self, index):
            data = dataset[index]
            time = self.sample_time()
            
            data["time"] = time

            return data
        
        dataset.__getitem__ = new_getitem

        return dataset


class AddBaseDistributionSampleToDataset():
    """
    A class that adds a sample from the base distribution to the data loaded by the dataset
    """

    def __init__(
            self,
            sample_method: Callable,
            relevant_parameter: str = "beta",
    ):
        """
        Args:
            sample_method: Callable: the method to sample from the base distribution
            relevant_parameter: str: the relevant parameter to sample from the base distribution
        """
        self.sample_method = sample_method
        self.relevant_parameter = relevant_parameter


    def __call__(self, dataset: torch.utils.data.Dataset) -> torch.utils.data.Dataset:
        """
        Add a sample from the base distribution to the dataset
        Args:
            dataset: torch.utils.data.Dataset: the dataset. The dataset yields a dictionary with the data
        Returns:
            torch.utils.data.Dataset: the dataset with the sample from the base distribution
        """
        
        def new_getitem(self, index):
            data = dataset[index]
            parameter_shape = data[self.relevant_parameter].shape

            sample = self.sample_method(parameter_shape)

            data[f"BaseDistributionSample_{self.relevant_parameter}"] = sample

            return data
        
        dataset.__getitem__ = new_getitem

        return dataset