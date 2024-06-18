import torch 

from PFNExperiments.LinearRegression.GenerativeModels.GenerateDataCurriculum import GenerateDataCurriculum, SyntheticDataCurriculumBatched
from PFNExperiments.LinearRegression.GenerativeModels.EpochLoader import EpochLoader
from typing import Tuple


"""
Functions and classes to generate data for conditional flow matching
"""
def sample_uniform_time() -> float:	
        """
        Just sample a uniform time point
        """
        return torch.rand(1).item()
    
def standard_normal_sample(shape: torch.Size) -> torch.Tensor:
    """
    Sample from the standard normal distribution
    Args:
        shape: torch.Size: the shape of the tensor to sample
    Returns:
        torch.Tensor: the sample
    """
    return torch.randn(shape)


class GenerateDataCurriculumCFM(GenerateDataCurriculum):
    """
    A class that generates data for conditional flow matching
    """

    def __init__(self,
                 time_sampling: callable = sample_uniform_time,
                 base_distribution_sampling: callable = standard_normal_sample,
                 relevant_parameter: str = "beta",
                 **kwargs,
                ):
        """
        Args:
            dataset_modifiers: list[callable]: the dataset modifiers
        """
        super(GenerateDataCurriculumCFM, self).__init__(**kwargs)
        self.time_sampling = time_sampling
        self.base_distribution_sampling = base_distribution_sampling
        self.relevant_parameter = relevant_parameter

       



    def make_dataloaders_for_epoch_dynamic(self,
                                 epoch:int = 0,
                                 n:int = 100,
                                 p:int = 5,
                                 n_samples_per_epoch:int = 10_000,
                                 batch_size:int = 256,
                                 train_frac = 0.7,
                                 val_frac = 0.15,
                                 shuffle: bool = True,
                                 use_seed: bool = False,
                                 n_samples_to_generate_at_once:int = 10_000
                                 ) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        """
        Make a dataloader where the data is generated on the fly
        Args:
            epoch: int: the epoch
            n: int: the number of observations per batch 
            p: int: the number of covariates
            n_samples_per_epoch : int: the number of samples per epoch
            batch_size: int: the batch size
            train_frac: float: the fraction of the data to use for training
            val_frac: float: the fraction of the data to use for validation
            shuffle: bool: whether to shuffle the data
            use_seed: bool: whether to use the seed for the random number generator
            n_samples_to_generate_at_once: int: the number of samples to generate at once
        Returns:
            Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]: a tuple of dataloaders for the training, test and validation data
        """


        train_size = int(train_frac * n_samples_per_epoch)
        val_size = int(val_frac * n_samples_per_epoch)
        test_size = n_samples_per_epoch - train_size - val_size

        if use_seed:
            seed = self.seed
        else:
            seed = torch.randint(0, 100000, (1,)).item()

        self.seed = seed

        
        dataset_train = SyntheticDataCurriculumBatchedCFM(n = n, 
                                p = p, 
                                n_samples_per_epoch = train_size, 
                                epoch = epoch,
                                pprogram_maker = self.pprogram_maker,
                                curriculum = self.curriculum,
                                pprogram_covariates = self.pprogram_covariates,
                                seed = None,
                                n_samples_to_generate_at_once = n_samples_to_generate_at_once,
                                time_sampling = self.time_sampling,
                                base_distribution_sampling = self.base_distribution_sampling,
                                relevant_parameter = self.relevant_parameter)
        
        
        dataset_val = SyntheticDataCurriculumBatchedCFM(n = n,
                                p = p,
                                n_samples_per_epoch = val_size,
                                epoch = epoch,
                                pprogram_maker = self.pprogram_maker,
                                curriculum = self.curriculum,
                                pprogram_covariates = self.pprogram_covariates,
                                seed = None,
                                n_samples_to_generate_at_once = n_samples_to_generate_at_once,
                                time_sampling = self.time_sampling,
                                base_distribution_sampling = self.base_distribution_sampling,
                                relevant_parameter = self.relevant_parameter)
        
        dataset_test = SyntheticDataCurriculumBatchedCFM(n = n,
                                p = p,
                                n_samples_per_epoch = test_size,
                                epoch = epoch,
                                pprogram_maker = self.pprogram_maker,
                                curriculum = self.curriculum,
                                pprogram_covariates = self.pprogram_covariates,
                                seed = None, 
                                n_samples_to_generate_at_once = n_samples_to_generate_at_once,
                                time_sampling = self.time_sampling,
                                base_distribution_sampling = self.base_distribution_sampling,
                                relevant_parameter = self.relevant_parameter)
        
        

        
        dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size = batch_size, shuffle = shuffle)
        dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size = batch_size, shuffle = shuffle)
        dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size = batch_size, shuffle = shuffle)

        return dataloader_train, dataloader_val, dataloader_test
    

    def make_epoch_loader(
            self, 
            n:int = 100,
            p:int = 5,
            number_of_batches_per_epoch:int = 10_000,
            n_epochs:int = 100,
            batch_size:int = 256,
            train_frac = 0.7,
            val_frac = 0.15,
            shuffle: bool = False,
            n_samples_to_generate_at_once:int = 10_000
            ) -> EpochLoader:
        """
        Make a loader where for each epoch the data is generated on the fly
        Args:
            n: int: the number of observations per batch 
            p: int: the number of covariates
            number_of_batches_per_epoch : int: the number of batches
            n_epochs: int: the number of epochs
            batch_size: int: the batch size
            train_frac: float: the fraction of the data to use for training
            val_frac: float: the fraction of the data to use for validation
            shuffle: bool: whether to shuffle the data
        
        Returns:
            EpochLoader: an epoch loader
        """

        if not abs(number_of_batches_per_epoch * n_epochs * batch_size - self.curriculum.max_iter) < batch_size*n_epochs:
            print(f"The number of batches times the number of epochs must be equal to the total number of iterations in the curriculum. But got {number_of_batches_per_epoch * n_epochs * batch_size} and {self.curriculum.max_iter} respectively")

        epoch_loader = EpochLoader(
            GenerateDataCurriculum = self,
            n_epochs = n_epochs,
            n = n,
            p = p,
            n_batches_per_epoch= number_of_batches_per_epoch,
            batch_size = batch_size,
            train_frac = train_frac,
            val_frac = val_frac,
            shuffle = shuffle,
            n_samples_to_generate_at_once = n_samples_to_generate_at_once
        )
        
        
        return epoch_loader
    






class SyntheticDataCurriculumBatchedCFM(SyntheticDataCurriculumBatched):
    """
    A class that generates data for conditional flow matching using the batched curriculum
    """
    
    def __init__(self,
                time_sampling: callable = sample_uniform_time,
                base_distribution_sampling: callable = standard_normal_sample,
                relevant_parameter: str = "beta",
                **kwargs):
        """
        Args:
            time_sampling: callable: the function that samples the time
            base_distribution_sampling: callable: the function that samples from the base distribution
            relevant_parameter: str: the relevant parameter for the base distribution
        """

        super(SyntheticDataCurriculumBatchedCFM, self).__init__(**kwargs)
        self.time_sampling = time_sampling
        self.base_distribution_sampling = base_distribution_sampling
        self.relevant_parameter = relevant_parameter

    def __getitem__(self, index):
        """
        Get an item from the dataset
        Args:
            index: int: the index
        Returns:
            dict: the data
        """
        data = super(SyntheticDataCurriculumBatchedCFM, self).__getitem__(index)
        time = self.time_sampling()
        base_sample = self.base_distribution_sampling(data[self.relevant_parameter].shape)
        data["time"] = time
        data[f"base_sample_{self.relevant_parameter}"] = base_sample
        return data