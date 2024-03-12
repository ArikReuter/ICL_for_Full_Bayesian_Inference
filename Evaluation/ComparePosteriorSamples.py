import pyro 
import torch
from scipy.stats import ks_2samp
import ot

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def compare_basic_statistics(P: torch.tensor, Q:torch.tensor) -> dict:
    """
    A method that compares the basic statistics of two sets of samples
    Args:
        P: torch.tensor: the first set of samples
        Q: torch.tensor: the second set of samples
    Returns:
        dict: a dictionary containing the basic statistics of the two sets of samples
    """
    assert P.shape[1:] == Q.shape[1:], f"The shape of the samples does not match after the first dimension. Have {P.shape} and {Q.shape}"

    mean_stats = {
        "absolute_mean_diff" : (P.mean(dim=0) - Q.mean(dim=0)).abs().mean(),
        "squared_mean_diff" : ((P.mean(dim=0) - Q.mean(dim=0))**2).mean(),
        "P_mean": P.mean(dim=0),
        "Q_mean": Q.mean(dim=0),
    }

    std_stats = {
        "absolute_std_diff" : (P.std(dim=0) - Q.std(dim=0)).abs().mean(),
        "squared_std_diff" : ((P.std(dim=0) - Q.std(dim=0))**2).mean(),
        "P_std": P.std(dim=0),
        "Q_std": Q.std(dim=0),
    }

    q_025_stats = {
        "absolute_q_025_diff" : (torch.quantile(P, 0.025, dim=0) - torch.quantile(Q, 0.025, dim=0)).abs().mean(),
        "squared_q_025_diff" : ((torch.quantile(P, 0.025, dim=0) - torch.quantile(Q, 0.025, dim=0))**2).mean(),
        "P_q_025": torch.quantile(P, 0.025, dim=0),
        "Q_q_025": torch.quantile(Q, 0.025, dim=0),
    }

    q_075_stats = {
        "absolute_q_075_diff" : (torch.quantile(P, 0.975, dim=0) - torch.quantile(Q, 0.975, dim=0)).abs().mean(),
        "squared_q_075_diff" : ((torch.quantile(P, 0.975, dim=0) - torch.quantile(Q, 0.975, dim=0))**2).mean(),
        "P_q_075": torch.quantile(P, 0.975, dim=0),
        "Q_q_075": torch.quantile(Q, 0.975, dim=0),
    }

    iqr_stats = {
        "absolute_iqr_diff" : (torch.quantile(P, 0.75, dim=0) - torch.quantile(P, 0.25, dim=0) - (torch.quantile(Q, 0.75, dim=0) - torch.quantile(Q, 0.25, dim=0))).abs().mean(),
        "squared_iqr_diff" : ((torch.quantile(P, 0.75, dim=0) - torch.quantile(P, 0.25, dim=0) - (torch.quantile(Q, 0.75, dim=0) - torch.quantile(Q, 0.25, dim=0)))**2).mean(),
        "P_iqr": torch.quantile(P, 0.75, dim=0) - torch.quantile(P, 0.25, dim=0),
        "Q_iqr": torch.quantile(Q, 0.75, dim=0) - torch.quantile(Q, 0.25, dim=0),
    }

    return {
        "mean_stats": mean_stats,
        "std_stats": std_stats,
        "q_025_stats": q_025_stats,
        "q_075_stats": q_075_stats,
        "iqr_stats": iqr_stats
    }

def compare_covariance(P: torch.tensor, Q: torch.tensor) -> dict:
    """
    A method that compares the covariance of two sets of samples
    Args:
        P: torch.tensor: the first set of samples
        Q: torch.tensor: the second set of samples
    Returns:
        dict: a dictionary containing the covariance statistics of the two sets of samples
    """
    assert P.shape[1:] == Q.shape[1:], f"The shape of the samples does not match after the first dimension. Have {P.shape} and {Q.shape}"


    cov_P = torch.cov(P.T)
    cov_Q = torch.cov(Q.T)


    cov_stats = {
        "absolute_cov_diff" : (cov_P - cov_Q).abs().mean(),
        "squared_cov_diff" : ((cov_P - cov_Q)**2).mean(),
        "P_cov": cov_P,
        "Q_cov": cov_Q,
    }

    return {
        "cov_stats": cov_stats
    }

def compare_Wasserstein(P: torch.tensor, Q:torch.tensor, metric = 'euclidean') -> dict:
    """
    Compare the Wasserstein-2 distance between two sets of samples
    Args:
        P: torch.tensor: the first set of samples
        Q: torch.tensor: the second set of samples
    Returns:
        dict: a dictionary containing the Wasserstein-2 distance between the two sets of samples
    """

    M = ot.dist(P, Q, metric=metric)

    a = torch.ones(P.shape[0]) / P.shape[0]
    b = torch.ones(Q.shape[0]) / Q.shape[0]

    W2 = ot.emd2(a, b, M)

    return {f"Wasserstein_distance with metric {metric}": W2.item()}

def compare_KLD_Gaussian(P: torch.tensor, Q: torch.tensor) -> dict:
    """
    Compute metrics for P and Q assuming they are Gaussian distributions.
    Args:
        P: torch.tensor: the first set of samples.
        Q: torch.tensor: the second set of samples.
    Returns:
        dict: a dictionary containing the metrics for the two sets of samples.
    Note:
        This implementation assumes that P and Q have the same dimensionality and that
        their covariance matrices are not singular. Numerical issues may arise for
        ill-conditioned covariance matrices.
    """

    mu_P, cov_P = P.mean(dim=0), torch.cov(P.T)
    mu_Q, cov_Q = Q.mean(dim=0), torch.cov(Q.T)

    cov_Q_inv = torch.inverse(cov_Q)  # Compute once for efficiency
    KLD_Gaussian = 0.5 * (torch.trace(cov_Q_inv @ cov_P) + 
                          (mu_Q - mu_P).T @ cov_Q_inv @ (mu_Q - mu_P) - 
                          P.shape[1] + 
                          torch.logdet(cov_Q) - torch.logdet(cov_P))
    
    return {"KLD_Gaussian": KLD_Gaussian.item()}

def compare_marginals_Wasserstein(P: torch.tensor, Q: torch.tensor, p=1, seed=None) -> dict:
    """
    A method that compares the marginals of two sets of samples using the Wasserstein-1 distance,
    with subsampling to equalize the number of samples if necessary.
    
    Args:
        P: torch.tensor: the first set of samples.
        Q: torch.tensor: the second set of samples.
        p: int: the power of the Wasserstein distance.
        seed: Optional[int]: a seed for the random number generator for reproducibility.
        
    Returns:
        dict: a dictionary containing the Wasserstein distance of the marginals of the two sets of samples.
    """

    assert len(P.shape) == 1 and len(Q.shape) == 1, "The input tensors must be 1-dimensional"
    if seed is not None:
        torch.manual_seed(seed)  # For reproducibility

    # Subsample the larger set to match the size of the smaller set if they have different sizes.
    min_size = min(len(P), len(Q))
    if len(P) > min_size:
        indices = torch.randperm(len(P))[:min_size]
        P = P[indices]
    elif len(Q) > min_size:
        indices = torch.randperm(len(Q))[:min_size]
        Q = Q[indices]

    # Ensure both P and Q are sorted.
    P_sorted, Q_sorted = torch.sort(P), torch.sort(Q)

    # note that torch.sort returns a tuple of the sorted tensor and the indices of the sorted tensor
    P_sorted, Q_sorted = P_sorted.values, Q_sorted.values


    # Compute the p-th power of the absolute differences between sorted samples.
    differences_pth_power = torch.pow(torch.abs(P_sorted - Q_sorted), p)

    # Compute the Wasserstein-p distance.
    wasserstein_p_distance = torch.mean(differences_pth_power)

    return {f"Wasserstein_distance with p = {p}": wasserstein_p_distance.item()}

def compare_marginals_KL(P: torch.tensor, Q: torch.tensor, seed=None) -> dict:
    """
    A method that compares the marginals of two sets of samples using the Kullback-Leibler divergence,
    with subsampling to equalize the number of samples if necessary.
    
    Args:
        P: torch.tensor: the first set of samples.
        Q: torch.tensor: the second set of samples.
        seed: Optional[int]: a seed for the random number generator for reproducibility.
        
    Returns:
        dict: a dictionary containing the Kullback-Leibler divergence of the marginals of the two sets of samples.
    """
    assert len(P.shape) == 1 and len(Q.shape) == 1, "The input tensors must be 1-dimensional"
    if seed is not None:
        torch.manual_seed(seed)

    # Subsample the larger set to match the size of the smaller set if they have different sizes.
    min_size = min(len(P), len(Q))
    if len(P) > min_size:
        indices = torch.randperm(len(P))[:min_size]
        P = P[indices]
    elif len(Q) > min_size:
        indices = torch.randperm(len(Q))[:min_size]
        Q = Q[indices]

    # Ensure both P and Q are sorted.
    P_sorted, Q_sorted = torch.sort(P), torch.sort(Q)
    # note that torch.sort returns a tuple of the sorted tensor and the indices of the sorted tensor
    P_sorted, Q_sorted = P_sorted.values, Q_sorted.values

    # Compute the Kullback-Leibler divergence.
    kl_divergence = torch.nn.functional.kl_div(P_sorted, Q_sorted, reduction='batchmean')

    return {"KL_divergence": kl_divergence.item()}

def compare_marginals_KS(P: torch.Tensor, Q: torch.Tensor) -> dict:
    """
    A method that compares the marginals of two sets of samples using the Kolmogorov-Smirnov test,
    accepting PyTorch tensors as input.
    
    Args:
        P: torch.Tensor: the first set of samples.
        Q: torch.Tensor: the second set of samples.
        
    Returns:
        dict: a dictionary containing the KS statistic and p-value of the comparison between the two sets of samples.
    """
    assert len(P.shape) == 1 and len(Q.shape) == 1, "The input tensors must be 1-dimensional"

    P_np = P.detach().cpu().numpy()
    Q_np = Q.detach().cpu().numpy()
    
    # Perform the Kolmogorov-Smirnov test
    ks_statistic, p_value = ks_2samp(P_np, Q_np)
    
    return {"KS_statistic": ks_statistic, "p_value": p_value}

def compare_marginals_mean(P: torch.Tensor, Q: torch.Tensor) -> dict:
    """
    A method that compares the mean of two sets of samples.
    
    Args:
        P: torch.Tensor: the first set of samples.
        Q: torch.Tensor: the second set of samples.
        
    Returns:
        dict: a dictionary containing the mean of the two sets of samples.
    """
    return {"mean_P": P.mean().item(), "mean_Q": Q.mean().item()}

def compare_marginals_std(P: torch.Tensor, Q: torch.Tensor) -> dict:
    """
    A method that compares the standard deviation of two sets of samples.
    
    Args:
        P: torch.Tensor: the first set of samples.
        Q: torch.Tensor: the second set of samples.
        
    Returns:
        dict: a dictionary containing the standard deviation of the two sets of samples.
    """
    return {"std_P": P.std().item(), "std_Q": Q.std().item()}

def compare_marginals_median(P: torch.Tensor, Q: torch.Tensor) -> dict:
    """
    A method that compares the median of two sets of samples.
    
    Args:
        P: torch.Tensor: the first set of samples.
        Q: torch.Tensor: the second set of samples.
        
    Returns:
        dict: a dictionary containing the median of the two sets of samples.
    """
    return {"median_P": torch.median(P).item(), "median_Q": torch.median(Q).item()}

def compare_marginals(P: torch.Tensor, 
                          Q: torch.Tensor, 
                          methods: list[callable] = [compare_marginals_Wasserstein, 
                                                     compare_marginals_KL, 
                                                     compare_marginals_KS, 
                                                    compare_marginals_mean,
                                                    compare_marginals_std,
                                                    compare_marginals_median]) -> dict:
        
        """
        A method that compares the marginals of two sets of samples using a list of comparison methods.

        Args:
            P: torch.Tensor: the first set of samples.
            Q: torch.Tensor: the second set of samples.
            methods: list[callable]: a list of comparison methods to use.

        Returns:
            dict: a dictionary containing the results of the comparison methods.
        """

        assert len(P.shape[1:]) == len(Q.shape[1:]), f"The shape of the samples does not match after the first dimension. Have {P.shape} and {Q.shape}"

        results = {}
        for dim in range(P.shape[1]):
            results[f"Marginal_dim_{dim}"] = {}
            for method in methods:
                results[f"Marginal_dim_{dim}"].update(method(P[:, dim], Q[:, dim]))

        return results

def compare_all_metrics(P: torch.tensor, Q: torch.tensor, methods: list[callable] = [compare_basic_statistics, compare_covariance, compare_Wasserstein, compare_KLD_Gaussian, compare_marginals]) -> dict:
    """
    A method that compares two sets of samples using a list of comparison methods.
    Args:
        P: torch.tensor: the first set of samples
        Q: torch.tensor: the second set of samples
        methods: list[callable]: a list of comparison methods to use
    Returns:
        dict: a dictionary containing the results of the comparison methods
    """
    results = {}
    for method in methods:
        results.update(method(P, Q))
    return results

def marginal_plots_hist_parallel(P: torch.tensor, Q: torch.tensor):
    """
    A method that plots the marginals of two sets of samples.
    Args:
        P: torch.tensor: the first set of samples
        Q: torch.tensor: the second set of samples
    """
    

    P_df = pd.DataFrame(P.numpy(), columns=[f"Dim_{i}" for i in range(P.shape[1])])
    Q_df = pd.DataFrame(Q.numpy(), columns=[f"Dim_{i}" for i in range(Q.shape[1])])

    for i in range(P.shape[1]):
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        sns.histplot(P_df[f"Dim_{i}"], ax=ax[0])
        ax[0].set_title(f"Marginal dim {i} P")
        sns.histplot(Q_df[f"Dim_{i}"], ax=ax[1])
        ax[1].set_title(f"Marginal dim {i} Q")

        # also plot the means as vertical lines
        ax[0].axvline(P_df[f"Dim_{i}"].mean(), color='r', linestyle='--')
        ax[1].axvline(Q_df[f"Dim_{i}"].mean(), color='r', linestyle='--')
        
        plt.show()

def marginal_plots_kde_parallel(P: torch.tensor, Q: torch.tensor):
    """
    A method that plots the marginals of two sets of samples using kernel density estimation.
    Args:
        P: torch.tensor: the first set of samples
        Q: torch.tensor: the second set of samples
    """
    P_df = pd.DataFrame(P.numpy(), columns=[f"Dim_{i}" for i in range(P.shape[1])])
    Q_df = pd.DataFrame(Q.numpy(), columns=[f"Dim_{i}" for i in range(Q.shape[1])])

    for i in range(P.shape[1]):
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        sns.kdeplot(P_df[f"Dim_{i}"], ax=ax[0])
        ax[0].set_title(f"Marginal dim {i} P")
        sns.kdeplot(Q_df[f"Dim_{i}"], ax=ax[1])
        ax[1].set_title(f"Marginal dim {i} Q")

        # also plot the means as vertical lines
        ax[0].axvline(P_df[f"Dim_{i}"].mean(), color='r', linestyle='--')
        ax[1].axvline(Q_df[f"Dim_{i}"].mean(), color='r', linestyle='--')

        plt.show()

def marginal_plots_kde_together(P: torch.tensor, Q: torch.tensor):
    """
    A method that plots the marginals of two sets of samples using kernel density estimation.
    Plot the densities for each dimension of the samples in the same plot.
    Args:
        P: torch.tensor: the first set of samples
        Q: torch.tensor: the second set of samples
    """
    P_df = pd.DataFrame(P.numpy(), columns=[f"Dim_{i}" for i in range(P.shape[1])])
    Q_df = pd.DataFrame(Q.numpy(), columns=[f"Dim_{i}" for i in range(Q.shape[1])])

    n_dims = P.shape[1]
    fig, ax = plt.subplots(1, n_dims, figsize=(5*n_dims, 5))

    for i in range(n_dims):
        sns.kdeplot(P_df[f"Dim_{i}"], ax=ax[i], label="P", color='b')
        sns.kdeplot(Q_df[f"Dim_{i}"], ax=ax[i], label="Q", color='r')
        ax[i].set_title(f"Marginal dim {i}")
        ax[i].legend()

        # also plot the means as vertical lines
        ax[i].axvline(P_df[f"Dim_{i}"].mean(), color='b', linestyle='--', label="P mean")
        ax[i].axvline(Q_df[f"Dim_{i}"].mean(), color='r', linestyle='--', label="Q mean")

    plt.show()

def projected_sample_plot(P: torch.tensor, Q: torch.tensor, method: str = "PCA"):
    """
    A method that plots the marginals of two sets of samples using kernel density estimation.
    Args:
        P: torch.tensor: the first set of samples
        Q: torch.tensor: the second set of samples
        method: str: the method to use for dimensionality reduction. Options are "PCA" and "tSNE"
    """
    assert method in ["PCA", "tSNE"], "The method must be either PCA or tSNE"
    if method == "PCA":
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        P_reduced = pca.fit_transform(P)
        Q_reduced = pca.fit_transform(Q)
    else:
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2)
        P_reduced = tsne.fit_transform(P)
        Q_reduced = tsne.fit_transform(Q)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].scatter(P_reduced[:, 0], P_reduced[:, 1])
    ax[0].set_title("P")
    ax[1].scatter(Q_reduced[:, 0], Q_reduced[:, 1])
    ax[1].set_title("Q")
    plt.show()