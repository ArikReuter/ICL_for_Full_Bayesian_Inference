import torch 
import os

from PFNExperiments.LinearRegression.ComparisonModels.PosteriorComparisonModel import PosteriorComparisonModel 
from PFNExperiments.LinearRegression.Models.ModelToPosterior import ModelToPosterior
from PFNExperiments.Training.Trainer import batch_to_model_lm
from PFNExperiments.Evaluation.ComparePosteriorSamples import compare_all_metrics, plot

class ModelComparison():
    """"
    Class to compare a ModelPosterior (a PFN) and a PosteriorComparisonModel (a model used for comparison such as HMC or VI)
    """

    def __init__(self, 
                 modelposterior: ModelToPosterior,
                 comparison_model: PosteriorComparisonModel,
                 batch_to_model_function_modelposterior: callable = batch_to_model_lm,
                 n_samples_modelposterior: int = 500,
                 save_path: str = "."
                 ) -> None:
        
        """
        Args: 
            modelposterior: ModelPosterior: the model posterior
            comparison_model: PosteriorComparisonModel: the comparison model
            n_samples_modelposterior: int: the number of samples to draw from the model posterior
            n_samples_comparison_model: int: the number of samples to draw from the comparison model
            save_path: str: the path to save the results

        """
        
        self.modelposterior = modelposterior
        self.modelposterior.model.eval()

        self.comparison_model = comparison_model
        self.batch_to_model_function_modelposterior = batch_to_model_function_modelposterior

        self.n_samples_modelposterior = n_samples_modelposterior

        self.save_path =  save_path + "/ModelComparison/"

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    
    def get_samples_modelposterior(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        A method that returns samples from the model posterior
        Args:
            X: torch.Tensor: the covariates of shape (batch_size, n, p) or (n, p)
            y: torch.Tensor: the response variable of shape (batch_size, n) or (n,)
        Returns:
            torch.Tensor: the samples from the model posterior of shape (batch_size, n_samples_modelposterior, p) or (n_samples_modelposterior, p)
        """

        # if X and y have no batch dimension, add it 

        if len(X.shape) == 2 and len(y.shape) == 1:
            X = X.unsqueeze(0)
            y = y.unsqueeze(0)

        assert len(X) == len(y), f"X and y need to have the same length, but got x.shape = {X.shape} and y.shape = {y.shape}"

        batch = {
            "x": X,
            "y": y
        }

        x = self.batch_to_model_function_modelposterior(batch)
        posterior = self.modelposterior.input_to_posterior(x)
        samples = posterior.sample((self.n_samples_modelposterior,))
        samples = samples.squeeze()

        if len(samples.shape) == 3:
            samples = samples.permute(1, 0, 2)

        return samples
    
    def get_samples_comparison_model(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        A method that returns samples from the comparison model
        Args:
            X: torch.Tensor: the covariates of shape (n, p)
            y: torch.Tensor: the response variable of shape (n,)
        Returns:
            torch.Tensor: the samples from the comparison model of shape (n_samples_comparison_model, n, p)
        """
        assert len(X.shape) == 2 and len(y.shape) == 1, f"X and y need to have shape (n, p) and (n,) but got x.shape = {X.shape} and y.shape = {y.shape}"

        posterior_samples = self.comparison_model.sample_posterior(X, y)
        return posterior_samples["beta"]
    
    def compare_samples_metrics(self, X: torch.Tensor, y: torch.Tensor) -> dict:
        """
        A method that compares the samples from the model posterior and the comparison model
        Args:
            X: torch.Tensor: the covariates of shape (n, p)
            y: torch.Tensor: the response variable of shape (n,)
        Returns:
            dict: a dictionary containing the metrics
        """
        modelposterior_samples = self.get_samples_modelposterior(X, y)
        comparison_model_samples = self.get_samples_comparison_model(X, y)

        return compare_all_metrics(modelposterior_samples, comparison_model_samples)
    
    def compare_samples_plot(self, X: torch.Tensor, y: torch.Tensor) -> None:
        """
        A method that compares the samples from the model posterior and the comparison model
        Args:
            X: torch.Tensor: the covariates of shape (n, p)
            y: torch.Tensor: the response variable of shape (n,)
        Returns:
            None
        """
        modelposterior_samples = self.get_samples_modelposterior(X, y)
        comparison_model_samples = self.get_samples_comparison_model(X, y)

        plot(modelposterior_samples, comparison_model_samples)

    def compare_samples_metrics_and_plot(self, X: torch.Tensor, y: torch.Tensor) -> dict:
        """
        A method that compares the samples from the model posterior and the comparison model
        Args:
            X: torch.Tensor: the covariates of shape (n, p)
            y: torch.Tensor: the response variable of shape (n,)
        Returns:
            dict: a dictionary containing the metrics
        """
        modelposterior_samples = self.get_samples_modelposterior(X, y)
        comparison_model_samples = self.get_samples_comparison_model(X, y)

        plot(modelposterior_samples, comparison_model_samples)

        return compare_all_metrics(modelposterior_samples, comparison_model_samples)
    
    def compare_sample_metrics_batch(self, X: torch.Tensor, y: torch.Tensor, plot:bool = True) -> (list[dict], dict, dict):
        """
        A method that compares the samples form the model posterior and the comparison model for a batch of data 
        Args:
            X: torch.Tensor: the covariates of shape (batch_size, n, p)
            y: torch.Tensor: the response variable of shape (batch_size, n)
            plot: bool: whether to plot the samples
        Returns:
            list[dict]: a list of dictionaries containing the metrics for each sample
            dict: a dictionary containing the metrics for the batch aggregated into the mean and std per metric
            dict: a dictionary containing the std per metric
        """
        batch_metrics = []
        for x, y in zip(X, y):
            if plot == True:
                metric_scores = self.compare_samples_metrics_and_plot(x, y)
            else:
                metric_scores = self.compare_samples_metrics(x, y)
            batch_metrics.append(metric_scores)

        batch_metrics_avg = {}
        for key1 in batch_metrics[0].keys():
            if isinstance(batch_metrics[0][key1], dict):
                batch_metrics_avg[key1] = {}
                for key2 in batch_metrics[0][key1].keys():
                    if isinstance(batch_metrics[0][key1][key2], float):
                        batch_metrics_avg[key1][key2] = torch.mean(torch.tensor([metric[key1][key2] for metric in batch_metrics]))
                    elif isinstance(batch_metrics[0][key1][key2], torch.Tensor):
                        if len(batch_metrics[0][key1][key2].shape) == 0:
                            batch_metrics_avg[key1][key2] = torch.mean(torch.tensor([metric[key1][key2] for metric in batch_metrics]))
                        else:
                            batch_metrics_avg[key1][key2] = torch.mean(torch.cat([metric[key1][key2] for metric in batch_metrics]))

            elif isinstance(batch_metrics[0][key1], float):
                batch_metrics_avg[key1] = torch.mean(torch.tensor([metric[key1] for metric in batch_metrics]))
                
            elif isinstance(batch_metrics[0][key1], torch.Tensor):
                batch_metrics_avg[key1] = torch.mean(torch.tensor([metric[key1] for metric in batch_metrics]))
                if len(batch_metrics_avg[key1].shape) == 0:
                    batch_metrics_avg[key1] = torch.mean(torch.tensor([metric[key1] for metric in batch_metrics]))
                else:
                    batch_metrics_avg[key1] = torch.mean(torch.cat([metric[key1] for metric in batch_metrics]))


        batch_metrics_std = {}
        for key1 in batch_metrics[0].keys():
            if isinstance(batch_metrics[0][key1], dict):
                batch_metrics_std[key1] = {}
                for key2 in batch_metrics[0][key1].keys():
                    if isinstance(batch_metrics[0][key1][key2], float):
                        batch_metrics_std[key1][key2] = torch.std(torch.tensor([metric[key1][key2] for metric in batch_metrics]))
                    elif isinstance(batch_metrics[0][key1][key2], torch.Tensor):
                        if len(batch_metrics[0][key1][key2].shape) == 0:
                            batch_metrics_std[key1][key2] = torch.std(torch.tensor([metric[key1][key2] for metric in batch_metrics]))
                        else:
                            batch_metrics_std[key1][key2] = torch.std(torch.cat([metric[key1][key2] for metric in batch_metrics]))

            elif isinstance(batch_metrics[0][key1], float):
                batch_metrics_std[key1] = torch.std(torch.tensor([metric[key1] for metric in batch_metrics]))
                
            elif isinstance(batch_metrics[0][key1], torch.Tensor):
                batch_metrics_std[key1] = torch.std(torch.tensor([metric[key1] for metric in batch_metrics]))
                if len(batch_metrics_std[key1].shape) == 0:
                    batch_metrics_std[key1] = torch.std(torch.tensor([metric[key1] for metric in batch_metrics]))
                else:
                    batch_metrics_std[key1] = torch.std(torch.cat([metric[key1] for metric in batch_metrics]))

        torch.save(batch_metrics, self.save_path + "batch_metrics.pt")
        torch.save(batch_metrics_avg, self.save_path + "batch_metrics_avg.pt")
        torch.save(batch_metrics_std, self.save_path + "batch_metrics_std.pt")

        return batch_metrics, batch_metrics_avg, batch_metrics_std