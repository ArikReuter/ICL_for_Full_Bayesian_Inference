import torch
import torch.utils 
from tqdm import tqdm

from PFNExperiments.LinearRegression.ComparisonModels.PosteriorComparisonModel import PosteriorComparisonModel 
from PFNExperiments.Evaluation.CompareModelToGT import convert_to_batchsize_1_dataloader
from PFNExperiments.Evaluation.CompareModelToGT import results_dict_to_data_x_y, results_dict_to_latent_variable_beta

from PFNExperiments.Evaluation.CompareModelToGT import CompareModelToGT
from PFNExperiments.Evaluation.CompareTwoModels import CompareTwoModels




def results_dict_to_data_x_y_tuple(result:dict) -> (torch.tensor, torch.tensor):
    """
    Take the dictionary with results and return the data x and y
    """
    x = result["x"]
    y = result["y"]

    x = x.squeeze(0)
    y = y.squeeze(0)
    return x, y

class Evaluate:
    """
    Class to perform evaluation of the probabilistic model
    """

    def __init__(
            self,
            posterior_model: PosteriorComparisonModel,
            evaluation_loader: torch.utils.data.DataLoader,
            comparison_models: list[PosteriorComparisonModel] = [],
            n_evaluation_cases: int = 100,
            #n_posterior_samples: int = 1000,
            results_dict_to_latent_variable: callable = results_dict_to_latent_variable_beta,
            results_dict_to_data_for_model: callable = results_dict_to_data_x_y_tuple,
            compare_to_gt: CompareModelToGT = CompareModelToGT(),
            compare_two_models: CompareTwoModels = CompareTwoModels(),
            verbose = True,
            compare_comparison_models_among_each_other = True
            
    ):
        """
        Args:
            posterior_model: PosteriorComparisonModel: the posterior model to evaluate
            evaluation_loader: torch.utils.data.DataLoader: the dataloader used for evaluation
            comparison_models: list[PosteriorComparisonModel]: the comparison models
            n_evaluation_cases: int: the number of evaluation cases to use, corresponds to the number of datasets to evaluate on
            n_posterior_samples: int: the number of posterior samples to draw from the posterior model and the comparison models
            results_dict_to_latent_variable: callable: a function that takes the results dictionary and returns the latent variable
            results_dict_to_data_for_model: callable: a function that takes the results dictionary and returns the data
            compare_to_gt: CompareModelToGT: the class to compare the model to the ground truth
            compare_two_models: CompareTwoModels: the class to compare two models
            verbose: bool: whether to print the results
            compare_comparison_models_among_each_other: bool: whether to compare the comparison models among each other
        """

        self.posterior_model = posterior_model
        self.evaluation_loader = convert_to_batchsize_1_dataloader(evaluation_loader)
        self.comparison_models = comparison_models
        self.n_evaluation_cases = n_evaluation_cases
        #self.n_posterior_samples = n_posterior_samples
        self.results_dict_to_latent_variable = results_dict_to_latent_variable
        self.results_dict_to_data_for_model = results_dict_to_data_for_model
        self.compare_to_gt = compare_to_gt
        self.compare_two_models = compare_two_models
        self.verbose = verbose
        self.compare_comparison_models_among_each_other = compare_comparison_models_among_each_other

        self.evaluation_list = self._convert_dataloader_samples_to_list()
        self.evaluation_list_alternative = self._convert_dataloader_samples_to_list()

    def _convert_dataloader_samples_to_list(self) -> list[dict]:
        """
        Convert the dataloader samples to a list of dictionaries
        Returns:
            list[dict]: a list of dictionaries
        """
        evaluation_list_data = []
        for i, d in enumerate(self.evaluation_loader):
            evaluation_list_data.append(d)
            if i == self.n_evaluation_cases:
                break
        return evaluation_list_data

    def sample_posterior_model(self, model: PosteriorComparisonModel) -> list[dict]:
        """
        Sample the posterior model
        Args:
            model: PosteriorComparisonModel: the model to sample from
        Returns:
            list[dict]: a list of dictionaries containing the posterior samples
        """

        posterior_samples = []
        for case in tqdm(self.evaluation_list, desc="Sampling posterior"):
            data = self.results_dict_to_data_for_model(case)
            samples = model.sample_posterior(*data)
            
            for key in case.keys():
                if key not in samples.keys():
                    samples[key] = case[key]
            
            posterior_samples.append(samples)

        return posterior_samples
    

    def _run_eval_raw_results(self) -> dict:
        """
        Run the evaluation
        Returns:
            dict: a dictionary containing the results
        """

        posterior_model_samples = self.sample_posterior_model(self.posterior_model)
        comparison_model_samples = [self.sample_posterior_model(model) for model in self.comparison_models]

        posterior_model_vs_gt = {(str(self.posterior_model), "gt"): self.compare_to_gt.compare(
            ground_truth_data1=self.evaluation_list,
            ground_truth_data2=self.evaluation_list_alternative,
            model_samples=posterior_model_samples
        ) }


        comparison_models_vs_gt = {
            (str(model), "gt"): self.compare_to_gt.compare(
            ground_truth_data1=self.evaluation_list,
            ground_truth_data2=self.evaluation_list_alternative,
            model_samples=model_samples
        ) for model, model_samples in zip(self.comparison_models, comparison_model_samples)
        }

        posterior_model_vs_comparison_models = {
            (str(self.posterior_model), str(model)): self.compare_two_models.compare_model_samples(posterior_model_samples, model_samples) for model, model_samples in zip(self.comparison_models, comparison_model_samples)
        }

        if self.compare_comparison_models_among_each_other:
            comparison_models_vs_comparison_models = {}
            for i in range(len(self.comparison_models)):
                for j in range(i+1, len(self.comparison_models)):
                    comparison_models_vs_comparison_models[(str(self.comparison_models[i]), str(self.comparison_models[j]))] = self.compare_two_models.compare_model_samples(comparison_model_samples[i], comparison_model_samples[j])
                

        res = {
            "posterior_model_vs_gt": posterior_model_vs_gt,
            "comparison_models_vs_gt": comparison_models_vs_gt,
            "posterior_model_vs_comparison_models": posterior_model_vs_comparison_models
        }

        if self.compare_comparison_models_among_each_other: 
            res["comparison_models_vs_comparison_models"] = comparison_models_vs_comparison_models

        return res

        
