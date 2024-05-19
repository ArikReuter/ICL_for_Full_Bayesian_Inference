import torch 
import pyro
from tqdm import tqdm
from typing import List, Dict, Tuple
import inspect
try:
    from LM_abstract import pprogram_linear_model_return_dict, return_only_y, pprogram_X
    from GenerateX import simulate_X_uniform
    from Curriculum import Curriculum
except:
    from PFNExperiments.LinearRegression.GenerativeModels.LM_abstract import pprogram_linear_model_return_dict, return_only_y, pprogram_X
    from PFNExperiments.LinearRegression.GenerativeModels.GenerateX import simulate_X_uniform


def get_function_parameter_names(func):
    # Get the signature of the function
    sig = inspect.signature(func)
    # Extract the parameter names from the signature
    param_names = list(sig.parameters.keys())
    return param_names


class GenerateDataCurriculum:
    """
    Class to generate data for (linear) regression with a curriculum
    Attributes:
        pprogram_maker: callable: a function that returns a probabilistic program
        curriculum: Curriculum: the curriculum that determines the way the samples are generated during training
    """

    def __init__(self,
                 pprogram_maker: callable,
                 curriculum: Curriculum,
                 pprogram_covariates: pprogram_X = simulate_X_uniform,
                 seed: int = 42):
        """
        Args:
            pprogram_maker: callable: a function that returns a probabilistic program
            curriculum: Curriculum: the curriculum
            pprogram_covariates: pprogram_X = simulate_X_uniform: a function that returns the covariates
            seed: int: the seed for the random number generator, default 42
        """
        
        pprogram_maker_param_names = get_function_parameter_names(pprogram_maker)
        curriculum_param_names = get_function_parameter_names(curriculum)

        self.seed = seed

        assert pprogram_maker_param_names == curriculum_param_names, "The parameters of the curriculum and the pprogram_maker must match"

        self.pprogram_maker = pprogram_maker
        self.pprogram_covariates = pprogram_covariates
        self.curriculum = curriculum
        torch.manual_seed(seed)
        pyro.set_rng_seed(seed)


    def make_dataloaders_for_epoch_dynamic(self,
                                 epoch:int = 0,
                                 n:int = 100,
                                 p:int = 5,
                                 n_batch:int = 10_000,
                                 batch_size:int = 256,
                                 train_frac = 0.7,
                                 val_frac = 0.15,
                                 shuffle: bool = True
                                 ) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        """
        Make a dataloader where the data is generated on the fly
        Args:
            epoch: int: the epoch
            n: int: the number of observations per batch 
            p: int: the number of covariates
            n_batch : int: the number of batches
            batch_size: int: the batch size
            train_frac: float: the fraction of the data to use for training
            val_frac: float: the fraction of the data to use for validation
            shuffle: bool: whether to shuffle the data
        Returns:
            Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]: a tuple of dataloaders for the training, test and validation data
        """

        train_size = int(train_frac * n_batch)
        val_size = int(val_frac * n_batch)
        test_size = n_batch - train_size - val_size

        dataset_train = SyntheticDataCurriculum(n = n, 
                                p = p, 
                                n_batch = train_size, 
                                epoch = epoch,
                                pprogram_maker = self.pprogram_maker,
                                curriculum = self.curriculum,
                                pprogram_covariates = self.pprogram_covariates,
                                seed = self.seed)
        
        dataset_val = SyntheticDataCurriculum(n = n,
                                p = p,
                                n_batch = val_size,
                                epoch = epoch,
                                pprogram_maker = self.pprogram_maker,
                                curriculum = self.curriculum,
                                pprogram_covariates = self.pprogram_covariates,
                                seed = self.seed)   
        
        dataset_test = SyntheticDataCurriculum(n = n,
                                p = p,
                                n_batch = test_size,
                                epoch = epoch,
                                pprogram_maker = self.pprogram_maker,
                                curriculum = self.curriculum,
                                pprogram_covariates = self.pprogram_covariates,
                                seed = self.seed)
        
        dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size = batch_size, shuffle = shuffle)
        dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size = batch_size, shuffle = shuffle)
        dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size = batch_size, shuffle = shuffle)

        return dataloader_train, dataloader_val, dataloader_test
                                               

    def make_epoch_loader(
            self, 
            n:int = 100,
            p:int = 5,
            n_batch:int = 10_000,
            batch_size:int = 256,
            train_frac = 0.7,
            val_frac = 0.15,
            shuffle: bool = True
            ) -> callable:
        """
        Make a loader where for each epoch the data is generated on the fly
        Args:
            n: int: the number of observations per batch 
            p: int: the number of covariates
            n_batch : int: the number of batches
            batch_size: int: the batch size
            train_frac: float: the fraction of the data to use for training
            val_frac: float: the fraction of the data to use for validation
            shuffle: bool: whether to shuffle the data
        
        Returns:
            callable: a function that returns dataloaders for the training, test and validation data
        """

        def epoch_loader(epoch:int) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
            """
            Make a dataloader where the data is generated on the fly
            Args:
                epoch: int: the epoch
            Returns:
                Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]: a tuple of dataloaders for the training, validation and test data
            """
            
            dataloaders = self.make_dataloaders_for_epoch_dynamic(epoch = epoch,
                                    n = n,
                                    p = p,
                                    n_batch = n_batch,
                                    batch_size = batch_size,
                                    train_frac = train_frac,
                                    val_frac = val_frac,
                                    shuffle = shuffle)
            
            return dataloaders
        
        return epoch_loader

    

class SyntheticDataCurriculum(torch.utils.data.Dataset):
    """
    A class to represent synthetic data that is generated on the fly
    This class uses a curriculum to determine the way the samples are generated
    Note that sampling is always random and the same seed is used for all batches
    """
    def __init__(
                self,
                n:int = 100,
                p:int = 5,
                n_batch:int = 10_000,
                epoch:int = 0,
                pprogram_maker:callable = None,
                curriculum: Curriculum = None,
                pprogram_covariates: pprogram_X = simulate_X_uniform,
                seed:int = 42
                ):
        """
        a  torch.utils.data.Dataset that generates synthetic data on the fly
        Args:
            n: int: the number of observations per batch 
            p: int: the number of covariates
            n_batch: int: the number of batches. Needs to be constant for all epochs
            epoch: int: the epoch
            pprogram_maker: callable: a function that returns a probabilistic program
            curriculum: Curriculum: the curriculum that determines the way the samples are generated during training
            pprogram: pprogram_linear_model_return_dict: a linear model probabilistic program
            pprogram_covariates: pprogram_X: a probabilistic program that simulates covariates
            seed: int: the seed for the random number generator
        """
        self.n = n
        self.p = p
        self.n_batch = n_batch
        self.epoch = epoch
        self.pprogram_maker = pprogram_maker
        self.curriculum = curriculum
        self.pprogram_covariates = pprogram_covariates
        self.seed = seed

        torch.manual_seed(seed)
        pyro.set_rng_seed(seed)
    
    def __len__(self) -> int:
        return self.n_batch

    def __getitem__(self, idx) -> dict:
        total_iteration = self.epoch * self.n_batch + idx

        pprogram = self.pprogram_maker(**self.curriculum(total_iteration))


        x = self.pprogram_covariates(self.n, self.p)

        while True:
            lm_res = pprogram(x)
            # if anything is nan or inf, sample again
            if all([torch.isfinite(lm_res[key]).all() for key in lm_res.keys()]):
                break

        return lm_res