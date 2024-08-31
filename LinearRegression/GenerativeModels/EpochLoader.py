import torch

from typing import Tuple


class EpochLoader():
    """
    Class to load data for each epoch 
    """

    def __init__(self,
                 GenerateDataCurriculum,
                 n_epochs: int = 100,
                 n : int = 100,
                 p : int = 5,
                 n_batches_per_epoch : int = 10_000,
                 batch_size : int = 256,
                 train_frac : float = 0.7,
                 val_frac : float = 0.15,
                 shuffle : bool = True,
                 n_samples_to_generate_at_once : int = 10_000,
                ):
        """
        Args: 
            GenerateDataCurriculum: GenerateDataCurriculum: the data generator
            n_epochs: int: the number of epochs
            n: int: the number of observations per batch 
            p: int: the number of covariates
            n_batches_per_epoch: int: the number of batches per epoch
            batch_size: int: the batch size
            train_frac: float: the fraction of the data to use for training
            val_frac: float: the fraction of the data to use for validation
            shuffle: bool: whether to shuffle the data
            n_samples_to_generate_at_once: int: the number of samples to generate at once
        """

        self.GenerateDataCurriculum = GenerateDataCurriculum
        self.n_epochs = n_epochs
        self.n = n
        self.p = p
        self.n_batches_per_epoch = n_batches_per_epoch
        self.batch_size = batch_size
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.shuffle = shuffle
        self.n_samples_to_generate_at_once = n_samples_to_generate_at_once

    def __repr__(self) -> str:
        representation = f"""EpochLoader(
        GenerateDataCurriculum = {self.GenerateDataCurriculum},
        n_epochs = {self.n_epochs},
        n = {self.n},
        p = {self.p},
        n_batches_per_epoch = {self.n_batches_per_epoch},
        batch_size = {self.batch_size},
        train_frac = {self.train_frac},
        val_frac = {self.val_frac},
        shuffle = {self.shuffle},
        n_samples_to_generate_at_once = {self.n_samples_to_generate_at_once}
        )"""
        return representation

    def __getitem__(self, epoch:int) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        """
        Get the dataloaders for an epoch
        Args:
            epoch: int: the epoch
        Returns:
            Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]: a tuple of dataloaders for the training, validation and test data
        """
        dataoaders = self.GenerateDataCurriculum.make_dataloaders_for_epoch_dynamic(
                                epoch = epoch,
                                n = self.n,
                                p = self.p,
                                n_samples_per_epoch = self.n_batches_per_epoch * self.batch_size,
                                batch_size = self.batch_size,
                                train_frac = self.train_frac,
                                val_frac = self.val_frac,
                                shuffle = self.shuffle,
                                use_seed = True,
                                n_samples_to_generate_at_once = self.n_samples_to_generate_at_once
        )
                                
        return dataoaders
    
    def __len__(self) -> int:
        return self.n_epochs
    
    def __call__(self, epoch:int) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        return self.__getitem__(epoch)


