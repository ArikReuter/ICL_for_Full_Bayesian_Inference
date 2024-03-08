import torch 
import pyro
from tqdm import tqdm
from typing import List, Dict, Tuple
from LM_abstract import pprogram_linear_model_return_dict, return_only_y, pprogram_X
from GenerateX import simulate_X_uniform



class GenerateData:
    """
    Class to generate data for (linear) regression
    """

    def __init__(self,
                 pprogram: pprogram_linear_model_return_dict,
                 pprogram_covariates: pprogram_X = simulate_X_uniform,
                 seed:int = 42
                 ):
        """
        Constructor for GenerateData
        Args:
            pprogram: pprogram_linear_model_return_dict: a linear model probabilistic program
            ppriogram_covariates: pprogram_X: a probabilistic program that simulates covariates
            seed: int: the seed for the random number generator
        """
        self.pprogram = pprogram
        self.pprogram_covariates = pprogram_covariates
        self.seed = seed
        
        torch.manual_seed(seed)
        pyro.set_rng_seed(seed)


    def simulate_data(self, 
                     n:int = 100, 
                     p:int = 5, 
                     n_batch:int = 10_000
                    ) -> Tuple[list[dict], int]:
        """
        Simulate data for a linear model with the same beta for each batch.
        Discard samples where the linear model is not finite to avoid numerical issues.
        Args:
            n: int: the number of observations per batch 
            p: int: the number of covariates
            n_batch: int: the number of batches
        Returns:
            Tuple[list[dict], int]: a list of dictionaries containing the simulated data and the number of discarded samples
        """

        n_discarded = 0

        data = []
        for i in tqdm(list(range(n_batch))):
            x = self.pprogram_covariates(n,p)

            while True:
                lm_res = self.pprogram(x)

                # if anything is nan or inf, sample again
                if all([torch.isfinite(lm_res[key]).all() for key in lm_res.keys()]):
                    break
                else:
                    n_discarded += 1

            data.append(lm_res)

        if n_discarded > 0:
            print(f"Warning: {n_discarded} samples were discarded because the linear model was not finite.")

        return data, n_discarded
    
    def render_model(self, p:int = 10):
        """
        Render the model
        """
        Xt = self.pprogram_covariates(1, p)
        yt = return_only_y(self.pprogram)(Xt)

        r = pyro.render_model(self.pprogram, model_args=(Xt, yt), render_distributions=True)
        return r
    
    def check_model(self, p:int = 3, n:int = 100, n_batch:int = 1_000):
        """
        Check the model for a few batches
        Args: 
            p: int: the number of covariates
            n: int: the number of observations
            n_batch: int: the number of batches
        """

        sample_data, discarded = self.simulate_data(n, p, n_batch)

        print(f"Discarded {discarded} samples")
        r = check_and_plot_data(sample_data)

        return r



def check_data(data: List[Dict[str, torch.tensor]]) -> dict:
    """
    Check statistics about the simulated data
    Args:
        data: List[Dict[str, torch.tensor]]: the simulated data in the form of a list of dictionaries where each element of the list is a dictionary containing the data for one batch
    
    Returns:
        dict: a dictionary containing the means and variances of the simulated data
    """
    # check for non-finite values
    for i, d in enumerate(data):
        for key in d.keys():
            if torch.isfinite(d[key]).all():
                pass
            else:
                raise ValueError(f"Data point {i} has non-finite values for {key}")

    stacked_data = {key: torch.stack([d[key] for d in data]) for key in data[0].keys()}

    means = {key: torch.mean(stacked_data[key], dim=0) for key in stacked_data.keys()}
    variances = {key: torch.var(stacked_data[key], dim=0) for key in stacked_data.keys()}
    minimums = {key: torch.min(stacked_data[key], dim=0).values for key in stacked_data.keys()}
    maximums = {key: torch.max(stacked_data[key], dim=0).values for key in stacked_data.keys()}

    return {
        "means": means,
        "variances": variances,
        "minimums": minimums,
        "maximums": maximums
    }

def check_and_plot_data(data: List[Dict[str, torch.tensor]]) -> None:
    """
    Check statistics about the simulated data and plot the data
    Args:
        data: List[Dict[str, torch.tensor]]: the simulated data in the form of a list of dictionaries where each element of the list is a dictionary containing the data for one batch
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    stats = check_data(data)

    stacked_data = {key: torch.stack([d[key] for d in data]) for key in data[0].keys()} 

    stats = check_data(data)

    X_overall_mean = stats["means"]["x"].mean()
    X_overall_variance = stats["variances"]["x"].mean()
    X_overall_min = stats["minimums"]["x"].min()
    X_overall_max = stats["maximums"]["x"].max()

    y_overall_mean = stats["means"]["y"].mean()
    y_overall_variance = stats["variances"]["y"].mean()
    y_overall_min = stats["minimums"]["y"].min()
    y_overall_max = stats["maximums"]["y"].max()


    overall_agg_stats = {
        'X': {
            'mean': X_overall_mean,
            'variance': X_overall_variance,
            'min': X_overall_min,
            'max': X_overall_max
        },
        'y': {
            'mean': y_overall_mean,
            'variance': y_overall_variance,
            'min': y_overall_min,
            'max': y_overall_max
        },
        'beta': {
            'mean': stats["means"]["beta"],
            'variance': stats["variances"]["beta"],
            'min': stats["minimums"]["beta"],
            'max': stats["maximums"]["beta"]
        }
    }

    print(overall_agg_stats)

    # also print the statistics for the other keys in the dictionary
    for key in stacked_data.keys():
        if key not in ["x", "y", "beta"]:
            print(f"Statistics for {key}:")
            print(f"Mean: {stats['means'][key]}")
            print(f"Variance: {stats['variances'][key]}")
            print(f"Min: {stats['minimums'][key]}")
            print(f"Max: {stats['maximums'][key]}")
            print("\n")

    # plot the data
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    sns.histplot(stacked_data["y"].flatten(), ax=ax[0])
    ax[0].set_title("Histogram of y")
    sns.histplot(stacked_data["x"].flatten(), ax=ax[1])
    ax[1].set_title("Histogram of x")
    plt.show()

    # plot each of the beta values in a histogram 
    p = stacked_data["beta"].shape[1]
    fig, ax = plt.subplots(1, p, figsize=(15, 5))
    for i in range(p):
        sns.histplot(stacked_data["beta"][:, i], ax=ax[i])
        ax[i].set_title(f"Histogram of beta_{i}")
    
    plt.show()

    # also plot all other keys in the dictionary
    for key in stacked_data.keys():
        if key not in ["x", "y", "beta"]:
            data = stacked_data[key]

            if len(data.shape) == 1:
                fig, ax = plt.subplots(1, 1, figsize=(5, 5))
                sns.histplot(data, ax=ax)
                ax.set_title(f"Histogram of {key}")
                plt.show()
            else:
                p = data.shape[1]
                fig, ax = plt.subplots(1, p, figsize=(15, 5))
                for i in range(p):
                    print(data[:, i])
                    sns.histplot(data[:, i], ax=ax[i])
                    ax[i].set_title(f"Histogram of {key}_{i}")
                plt.show()

    return overall_agg_stats


