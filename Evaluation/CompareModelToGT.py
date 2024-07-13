import torch 
from tqdm import tqdm
from PFNExperiments.LinearRegression.ComparisonModels.PosteriorComparisonModel import PosteriorComparisonModel 

from PFNExperiments.Evaluation.BasicMetrics import compare_Wasserstein
from PFNExperiments.Evaluation.MMD import compare_samples_mmd
from PFNExperiments.Evaluation.ClassifcationBasedComparison import compare_samples_classifier_based

from PFNExperiments.Evaluation.PerplexityKDE import compare_to_gt_perplexity_kde
from PFNExperiments.Evaluation.CompareBasicStatisticsToGT import compare_to_gt_MAP, compare_to_gt_mean_difference


def convert_to_batchsize_1_dataloader(dataloader: torch.utils.data.DataLoader) -> torch.utils.data.DataLoader:
    """"
    Take a dataloader and return a new dataloader with a batch size of 1
    Args:
        dataloader: torch.utils.data.DataLoader: the dataloader to convert
    Returns:
        torch.utils.data.DataLoader: the new dataloader with a batch size of 1
    """
    return torch.utils.data.DataLoader(dataloader.dataset, batch_size=1, shuffle=False)


def results_dict_to_latent_variable_beta(result:dict) ->  torch.tensor:
    """
    Take the dictionary with results and return the latent variable
    """
    return result["beta"]

def results_dict_to_data_x_y(result:dict) -> (torch.tensor):
    """
    Take the dictionary with results and return the data x and y concatenated 
    """
    x = result["x"]
    y = result["y"]

    y = y.reshape(1, -1)
    X_y = torch.cat([x, y.unsqueeze(-1)], dim = -1) # concatenate the x and y values to one data tensor
    return X_y

def flatten_dict_list(input_list: list[dict]) -> dict:
    """
    Flatten a list of dictionaries
    Args:
        input_list: list[dict]: the list of dictionaries to flatten
    
    Returns:
        dict: the flattened dictionary
    """
    new_dict_list = []

    for item in input_list:
        new_dict = {}
        for outer_key, inner_dict in item.items():
            for inner_key, value in inner_dict.items():
                new_dict[inner_key] = value

        new_dict_list.append(new_dict)



    return new_dict_list
    
def try_otherwise_return_error(fun: callable) -> callable:
    """
    A wrapper that modifies a fuction to return torch.nan if the function fails
    Args:
        fun: callable: the function
    Returns:
        callable: a function that returns torch.nan if the original function fails
    """
    def wrapper(*args, **kwargs):
        try:
            return fun(*args, **kwargs)
        except Exception as e:
            return str(e)
    return wrapper

class CompareModelToGT():
    """
    A class to compare the model to the ground truth samples
    """

    def __init__(self,
                 results_dict_to_latent_variable: callable = results_dict_to_latent_variable_beta,
                 results_dict_to_data: callable = results_dict_to_data_x_y,
                 metrics_joint: dict =  {
                        "Wasserstein Metric": try_otherwise_return_error(compare_Wasserstein),
                        "MMMD": try_otherwise_return_error(compare_samples_mmd),
                        "CST": try_otherwise_return_error(compare_samples_classifier_based)
                 },
                n_draws_posterior_to_sample_joint: int = 100,
                metrics_gt_parameter: dict = {
                        "Perplextiy": try_otherwise_return_error(compare_to_gt_perplexity_kde),
                        "MAP diff": try_otherwise_return_error(compare_to_gt_MAP),
                        "Mean diff": try_otherwise_return_error(compare_to_gt_mean_difference)
                },

                    ) -> None:
        
        """
        Args:
            results_dict_to_latent_variable: callable: a function that takes the results dictionary and returns the latent variable
            results_dict_to_data: callable: a function that takes the results dictionary and returns the data
            metrics_joint: dict: a dictionary of metrics to compare the model samples to the ground truth samples
            n_draws_poterior_to_sample_joint: int: the number of draws from the posterior to sample the joint distribution. there can be serveral those draws because for the model we have several posterior samples per data point
            metrics_gt_parameter: dict: a dictionary of metrics to compare the model samples to the ground truth parameters
        """
        
        self.results_dict_to_latent_variable = results_dict_to_latent_variable
        self.results_dict_to_data = results_dict_to_data

        self.metrics_joint = metrics_joint
        self.n_draws_posterior_to_sample_joint = n_draws_posterior_to_sample_joint
        self.metrics_gt_parameter = metrics_gt_parameter

    def compare_joint_samples(
                self,
                ground_truth_data: list[dict],
                model_samples: list[dict]
                 ) -> dict:
        """
        A method that compares the model samples to the ground truth samples
        Args:
            ground_truth_data: list[dict]: a list of dictionaries containing the ground truth data. 
            model_samples: list[dict]: a list of dictionaries containing the model samples
        Returns:
            list[dict]: a list of dictionaries containing the results of the comparison
        """ 

        assert len(ground_truth_data) == len(model_samples), "The number of ground truth samples and model samples should match. but got {} and {}".format(len(ground_truth_data), len(model_samples))

        result = []

        for _ in tqdm(list(range(self.n_draws_posterior_to_sample_joint))):  # since we have multiple samples from the posterior of the model, we can compare each of them to the ground truth samples several times
            model_latent_variable_sample_list = []
            gt_latent_variable_sample_list = []

            model_data_sample_list = [] 
            gt_data_sample_list = []
            
            for gt, model in zip(ground_truth_data, model_samples):
                gt_latent_variable = self.results_dict_to_latent_variable(gt)
                model_latent_variable = self.results_dict_to_latent_variable(model)

                gt_data = self.results_dict_to_data(gt)
                model_data = self.results_dict_to_data(model)

                assert not torch.any(gt_data == model_data), "The data for the ground truth and model samples should not match. but got {} and {}".format(gt_data, model_data) 

                # randomly pick a sample for the latent variable by the model 
                model_latent_variable_sample = model_latent_variable[torch.randint(0, model_latent_variable.shape[0], (1,))]

                model_latent_variable_sample_list.append(model_latent_variable_sample)
                gt_latent_variable_sample_list.append(gt_latent_variable)

                model_data_sample_list.append(model_data)
                gt_data_sample_list.append(gt_data)

            model_latent_variable_sample = torch.stack(model_latent_variable_sample_list).squeeze()
            gt_latent_variable = torch.stack(gt_latent_variable_sample_list).squeeze()

            model_data_sample = torch.stack(model_data_sample_list).squeeze().flatten(start_dim=1)
            gt_data = torch.stack(gt_data_sample_list).squeeze().flatten(start_dim=1)

        

            joint_samples_model = torch.cat([model_latent_variable_sample, model_data_sample], dim = -1)
            joint_samples_gt = torch.cat([gt_latent_variable, gt_data], dim = -1)

            r = {}

            #print(f"model_latent_variable_sample: {model_latent_variable_sample.shape}")
            #print(f"gt_latent_variable: {gt_latent_variable.shape}")

            for metric_name, metric in self.metrics_joint.items():
                r[metric_name] = metric(joint_samples_model, joint_samples_gt)

            result.append(r)

        try:
            result = flatten_dict_list(result)
        except:
            pass

        return result

    def compare_gt_parameters(
            self,
            ground_truth_data: list[dict],
            model_samples: list[dict]
            ) -> list[dict]:

        """
        Compare the model samples for the variables z to the ground truth value of z
        Args:
            ground_truth_data: list[dict]: a list of dictionaries containing the ground truth data. 
            model_samples: list[dict]: a list of dictionaries containing the model samples. Note that the ground truth data for both cases has to match to be able to compare the posterior. 
        Returns:
            list[dict]: a list of dictionaries containing the results of the comparison
        """
        assert len(ground_truth_data) == len(model_samples), "The number of ground truth samples and model samples should match. but got {} and {}".format(len(ground_truth_data), len(model_samples))  

        result = []

        for gt, model in tqdm(list(zip(ground_truth_data, model_samples))):
            gt_latent_variable = self.results_dict_to_latent_variable(gt)
            model_latent_variable = self.results_dict_to_latent_variable(model)

            assert gt_latent_variable.shape[0] == 1, "The model samples should only contain one sample. but got {}".format(model_latent_variable.shape[0])
     

            gt_data = self.results_dict_to_data(gt)
            model_data = self.results_dict_to_data(model)

            assert gt_data.shape[0] == 1, "The model samples should only contain one sample. but got {}".format(model_data.shape[0])

            assert torch.all(gt_data == model_data), "The data for the ground truth and model samples should match. but got {} and {}".format(gt_data, model_data)
            
            r = {}

            for metric_name, metric in self.metrics_gt_parameter.items():
                r[metric_name] = metric(model_latent_variable, gt_latent_variable)

            result.append(r)

        return result

    
    def compare(self, ground_truth_data1: list[dict], ground_truth_data2: list[dict], model_samples: list[dict]) -> dict:
        """
        Compare the model samples to the ground truth samples and parameters
        Args:
            ground_truth_data1: list[dict]: a list of dictionaries containing the ground truth data. Here the data part does not coincide with the data part of the model samples.
            ground_truth_data2: list[dict]: a list of dictionaries containing the ground truth data. Here the data part coincides with the data part of the model samples.
            model_samples: list[dict]: a list of dictionaries containing the model samples
        Returns:
            dict: a dictionary containing the results of the comparison
        """
        results = {}
        gt_paramter_results = self.compare_gt_parameters(ground_truth_data1, model_samples)
        joint_results = self.compare_joint_samples(ground_truth_data2, model_samples)
        

        joint_results2 = []
        # rename the keys of the joint results by adding "joint" to the key

        for elem in joint_results:
            new_d = {}
            for key, value in elem.items():
                new_d["joint_" + key] = value

            joint_results2.append(new_d)

        gt_paramter_results2 = []
        # rename the keys of the gt parameter results by adding "gt_parameter" to the key

        for elem in gt_paramter_results:
            new_d = {}
            for key, value in elem.items():
                new_d["gt_parameter_" + key] = value

            gt_paramter_results2.append(new_d)

        results["joint"] = joint_results2
        results["gt_parameter"] = gt_paramter_results2

        return results


