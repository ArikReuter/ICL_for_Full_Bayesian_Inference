import torch 

from PFNExperiments.LinearRegression.ComparisonModels.PosteriorComparisonModel import PosteriorComparisonModel 
from PFNExperiments.LinearRegression.Models.ModelToPosterior import ModelToPosterior
from PFNExperiments.Training.Trainer import batch_to_model_lm
from PFNExperiments.Evaluation.ComparePosteriorSamples import compare_all_metrics, plot

from PFNExperiments.LinearRegression.Evaluation.CompareModels import ModelComparison 


class CompareComparisonModels(ModelComparison):
    """
    Class analogous to ModelComparison but for comparison models
    """

    def __init__(self, 
                 comparison_model_1: PosteriorComparisonModel,
                 comparison_model_2: PosteriorComparisonModel,
                 n_samples_comparison_model_1: int = 500,
                 n_samples_comparison_model_2: int = 500
                 ) -> None:
        """
        Args: 
            comparison_model_1: PosteriorComparisonModel: the comparison model
            comparison_model_2: PosteriorComparisonModel: the comparison model
            n_samples_comparison_model_1: int: the number of samples to draw from the comparison model
            n_samples_comparison_model_2: int: the number of samples to draw from the comparison model

        """
        
        self.comparison_model_1 = comparison_model_1
        self.comparison_model_2 = comparison_model_2

        self.n_samples_comparison_model_1 = n_samples_comparison_model_1
        self.n_samples_comparison_model_2 = n_samples_comparison_model_2




    def compare_samples_metrics(self, X: torch.Tensor, y: torch.Tensor) -> dict:
        """
        A method that compares the samples from the model posterior and the comparison model
        Args:
            X: torch.Tensor: the covariates of shape (n, p)
            y: torch.Tensor: the response variable of shape (n,)
        Returns:
            dict: a dictionary containing the metrics
        """
        comparison_samples_1 = get_samples_comparison_model(X, y, self.comparison_model_1)
        comparison_samples_2 = get_samples_comparison_model(X, y, self.comparison_model_2)

        return compare_all_metrics(comparison_samples_1, comparison_samples_2)

    def compare_samples_plot(self, X: torch.Tensor, y: torch.Tensor) -> None:
        """
        A method that compares the samples from the model posterior and the comparison model
        Args:
            X: torch.Tensor: the covariates of shape (n, p)
            y: torch.Tensor: the response variable of shape (n,)
        Returns:
            None
        """
        comparison_samples_1 = get_samples_comparison_model(X, y, self.comparison_model_1)
        comparison_samples_2 = get_samples_comparison_model(X, y, self.comparison_model_2)

        plot(comparison_samples_1, comparison_samples_2)

    def compare_samples_metrics_and_plot(self, X: torch.Tensor, y: torch.Tensor) -> dict:
        """
        A method that compares the samples from the model posterior and the comparison model
        Args:
            X: torch.Tensor: the covariates of shape (n, p)
            y: torch.Tensor: the response variable of shape (n,)
        Returns:
            dict: a dictionary containing the metrics
        """
        comparison_samples_1 = get_samples_comparison_model(X, y, self.comparison_model_1)
        comparison_samples_2 = get_samples_comparison_model(X, y, self.comparison_model_2)

        plot(comparison_samples_1, comparison_samples_2)

        return compare_all_metrics(comparison_samples_1, comparison_samples_2)



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

        return batch_metrics, batch_metrics_avg, batch_metrics_std


def get_samples_comparison_model(X: torch.Tensor, y: torch.Tensor, comparison_model: PosteriorComparisonModel) -> torch.Tensor:
    """
    A method that returns samples from the comparison model
    Args:
        X: torch.Tensor: the covariates of shape (n, p)
        y: torch.Tensor: the response variable of shape (n,)
        comparison_model: PosteriorComparisonModel: the comparison model
    Returns:
        torch.Tensor: the samples from the comparison model of shape (n_samples_comparison_model, n, p)
    """
    assert len(X.shape) == 2 and len(y.shape) == 1, f"X and y need to have shape (n, p) and (n,) but got x.shape = {X.shape} and y.shape = {y.shape}"

    posterior_samples = comparison_model.sample_posterior(X, y)
    return posterior_samples["beta"].detach()