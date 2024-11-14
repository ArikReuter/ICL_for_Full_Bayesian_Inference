from PFNExperiments.Evaluation.RealWorldEvaluation.EvaluateRealWorld import EvaluateRealWorld, just_return_results, results_dict_to_data_x_y_tuple, results_dict_to_latent_variable_beta0_and_beta
import torch

from PFNExperiments.Evaluation.Evaluate import Evaluate
from IPython.display import display


from PFNExperiments.LinearRegression.ComparisonModels.PosteriorComparisonModel import PosteriorComparisonModel 

from PFNExperiments.Evaluation.CompareTwoModels import CompareTwoModels
from PFNExperiments.Evaluation.Plot import Plot

import pandas  as pd
import numpy as np
import pickle
import os
from scipy.stats import m


def compute_mse_posterior_expectation(
        y_true: torch.Tensor,
        y_samples: torch.Tensor
) -> float:
    """
    Compute the mean squared error of the posterior expectation to the true values
    Args:
        y_true: torch.Tensor: the true values (n_test_points,)
        y_samples: torch.Tensor: the samples from the posterior (n_samples, n_test_points)
    
    Returns:
        float: the mean squared error
    """

    assert y_true.dim() == 1, f"The true values must have dimension 1. But got {y_true.dim()}"
    assert y_samples.dim() == 2, f"The samples must have dimension 2. But got {y_samples.dim()}"
    assert y_true.shape[0] == y_samples.shape[1], f"The number of test points must be equal for the true values and the samples. But got {y_true.shape[0]} and {y_samples.shape[1]}"

    posterior_expectation = y_samples.mean(dim=0)  # shape (n_test_points,)

    mse = ((y_true - posterior_expectation) ** 2).mean().item()

    return mse


class EvaluatePredictive(EvaluateRealWorld):
    """
    A class to evaluate the predictive performance of a model. 
    """



    def __init__(
            self,
            posterior_model: PosteriorComparisonModel,
            include_intercept: bool,
            sample_noise_posterior_model: callable,
            evaluation_datasets: list[dict],
            comparison_models: list[PosteriorComparisonModel] = [],
            sample_noise_comparison_models: list[callable] = [],
            model_names: list[str] = None,
            n_evaluation_cases: int = 1,
            results_dict_to_latent_variable_posterior_model: callable = just_return_results,
            results_dict_to_latent_variable_comparison_models: callable = results_dict_to_latent_variable_beta0_and_beta,
            results_dict_to_data_for_model: callable = results_dict_to_data_x_y_tuple,
            result_dict_to_data_for_comparison_models: callable = None,
            compare_two_models: CompareTwoModels = CompareTwoModels(),
            verbose = True,
            compare_comparison_models_among_each_other = True,
            save_path: str = None,
            overwrite_results: bool = False
            
    ):
        """
        Args:
            posterior_model: PosteriorComparisonModel: the posterior model to evaluate
            include_intercept: bool: whether to include an intercept in the model
            sample_noise_posterior_model: callable: a function to sample additive noise for the posterior model
            evaluation_loader: torch.utils.data.DataLoader: the dataloader used for evaluation
            comparison_models: list[PosteriorComparisonModel]: the comparison models
            sample_noise_comparison_models: list[callable]: a list of functions to sample additive noise for the comparison models
            n_evaluation_cases: int: the number of evaluation cases to use, corresponds to the number of datasets to evaluate on
            n_posterior_samples: int: the number of posterior samples to draw from the posterior model and the comparison models
            results_dict_to_latent_variable: callable: a function that takes the results dictionary and returns the latent variable
            results_dict_to_data_for_model: callable: a function that takes the results dictionary and returns the data
            results_dict_to_data_for_model: callable: a function that takes the results dictionary and returns the data
            compare_to_gt: CompareModelToGT: the class to compare the model to the ground truth
            compare_two_models: CompareTwoModels: the class to compare two models
            verbose: bool: whether to print the results
            compare_comparison_models_among_each_other: bool: whether to compare the comparison models among each other
            save_path: str: the path to save the results
            overwrite_results: bool: whether to overwrite the results if they already exist
        """

        assert len(evaluation_datasets) >= n_evaluation_cases, f"The number of evaluation cases is larger than the number of datasets. But got {len(evaluation_datasets)} and {n_evaluation_cases}"


        self.posterior_model = posterior_model
        self.include_intercept = include_intercept
        self.sample_noise_posterior_model = sample_noise_posterior_model
        self.evaluation_list = evaluation_datasets[:n_evaluation_cases]
        self.comparison_models = comparison_models
        self.sample_noise_comparison_models = sample_noise_comparison_models
        self.n_evaluation_cases = n_evaluation_cases
        #self.n_posterior_samples = n_posterior_samples
        self.results_dict_to_latent_variable_posterior_model = results_dict_to_latent_variable_posterior_model
        self.results_dict_to_latent_variable_comparison_models = results_dict_to_latent_variable_comparison_models
        self.results_dict_to_data_for_model = results_dict_to_data_for_model
        self.compare_two_models = compare_two_models
        self.verbose = verbose
        self.compare_comparison_models_among_each_other = compare_comparison_models_among_each_other
        self.save_path = save_path
        self.overwrite_results = overwrite_results

        if result_dict_to_data_for_comparison_models is None:
            self.result_dict_to_data_for_comparison_models = self.results_dict_to_data_for_model
        else:
            self.result_dict_to_data_for_comparison_models = result_dict_to_data_for_comparison_models

        if model_names is None:
            comparison_models_names = [str(model) for model in comparison_models]
            model_names = [str(posterior_model)] + comparison_models_names

        assert len(model_names) == len(comparison_models) +1, f"The number of model names must be equal to the number of comparison models + 1. But got {len(model_names)} and {len(comparison_models) +1}"
        model_names_dict = {model: name for model, name in zip([posterior_model] + comparison_models, model_names)}
        self.model_names_dict = model_names_dict

        # check if the save path exists, if not create it. If it exists and is not empty, check if the overwrite flag is set

        if self.save_path is not None:
            # append "real_world" to save path
            

            if not os.path.exists(self.save_path):
                print(f"The save path {self.save_path} does not exist, creating it")
                os.makedirs(self.save_path)
            else:
                if len(os.listdir(self.save_path)) > 0:
                    if not self.overwrite_results:
                        raise ValueError("The save path is not empty and the overwrite flag is not set")
                    else:
                        print("The save path is not empty and the overwrite flag is set, the results will be overwritten")

        if self.save_path is not None:
            plot_save_path = f"{self.save_path}/plots" if self.save_path is not None else None
            if not os.path.exists(plot_save_path):
                os.makedirs(plot_save_path)

        else:
            plot_save_path = None
        self.plot = Plot(save_path=plot_save_path)


    def sample_posterior_predictive(
            self,
            coefficient_samples: torch.Tensor,
            noise_samples: torch.Tensor,
            X: torch.Tensor,
            inlcude_intercept: bool = True
    ):
        """
        Compute samples from the posterior predictive distribution
        Args:
            coefficient_samples: torch.Tensor: the coefficient samples (n_samples, n_features) or (n_samples, n_features + 1) if the intercept is included
            noise_samples: torch.Tensor: the noise samples (n_samples, n_test_points)
            X: torch.Tensor: the input data (n_test_points, n_features)

        Returns:
            torch.Tensor: the predicted means (n_test_points, n_samples)
        """

        n_samples, n_features = coefficient_samples.shape
        n_test_points = X.shape[0]

        assert noise_samples.shape == (n_samples, n_test_points), f"The noise samples must have shape (n_samples, n_test_points). But got {noise_samples.shape}"
        if not inlcude_intercept:
            assert n_features == X.shape[1], f"The number of features in the coefficient samples must be equal to the number of features in X. But got {n_features} and {X.shape[1]}"
        else:
            assert n_features -1 == X.shape[1], f"The number of features in the coefficient samples must be equal to the number of features in X. But got {n_features} and {X.shape[1]}"
        
        if inlcude_intercept:
            X = torch.cat([torch.ones(n_test_points, 1), X], dim=1)

        predicted_means = X @ coefficient_samples.T  # shape (n_test_points, n_samples)
        
        predictions = predicted_means + noise_samples.T  # shape (n_test_points, n_samples)

        return predictions



    def _run_eval_raw_results(self) -> tuple:
        """
        Run the evaluation
        Returns:
            dict: a dictionary containing the results in form of dataframes
            dict: a dictionary containing the results in form of raw results
        """

      
        posterior_model_samples = self.sample_posterior_model(self.posterior_model, is_comparison_model=False)
        comparison_model_samples = [self.sample_posterior_model(model, is_comparison_model=True) for model in self.comparison_models]

        #print(posterior_model_samples)
      
        self.posterior_model_samples = posterior_model_samples
        self.comparison_model_samples = comparison_model_samples

        for i in range(self.n_evaluation_cases):
            evaluation_data = self.evaluation_list[i]
            x_test = evaluation_data["x_test"]
            y_test = evaluation_data["y_test"]

            posterior_model_samples_coefficients = posterior_model_samples[i]
            posterior_model_samples_noise = self.sample_noise_posterior_model(y_test.shape[0] * posterior_model_samples_coefficients.shape[0]).reshape(posterior_model_samples_coefficients.shape[0], -1)
            posterior_model_predictions = self.sample_posterior_predictive(posterior_model_samples_coefficients, posterior_model_samples_noise, x_test, self.include_intercept)

            comparison_model_samples_coefficients = [comparison_model_samples[i] for comparison_model_samples in comparison_model_samples]
            comparison_model_samples_noise = [sample_noise_comparison_model(y_test.shape[0] * comparison_model_samples_coefficients[0].shape[0]).reshape(comparison_model_samples_coefficients[0].shape[0], -1) 
                                              for sample_noise_comparison_model in self.sample_noise_comparison_models]
            comparison_model_predictions = [self.sample_posterior_predictive(coefficient_samples, noise_samples, x_test, self.include_intercept) 
                                            for coefficient_samples, noise_samples in zip(comparison_model_samples_coefficients, comparison_model_samples_noise)]

            # compute the mse of the posterior expectation
            mse_posterior_expectation = compute_mse_posterior_expectation(y_test, posterior_model_predictions)

            # compute the mse of the comparison models

            mse_comparison_models = [compute_mse_posterior_expectation(y_test, comparison_model_prediction) for comparison_model_prediction in comparison_model_predictions]

            return {
                "mse_posterior_model": mse_posterior_expectation,
                "mse_comparison_models": mse_comparison_models
            }