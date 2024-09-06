from PFNExperiments.LinearRegression.GenerativeModels.GenerateDataLM_Examples import make_lm_program_ig_batched, make_lm_program_ig
from PFNExperiments.LinearRegression.GenerativeModels.GenerateDataLM_ExamplesIntercept import make_lm_program_ig_intercept_batched, make_lm_program_ig_intercept
from PFNExperiments.LinearRegression.GenerativeModels.GenerateDataLM_ExamplesSparsity import make_lm_program_Laplace_ig_batched, make_lm_program_Laplace_ig, make_lm_program_Laplace_ig_intercept_batched, make_lm_program_Laplace_ig_intercept
from PFNExperiments.LinearRegression.GenerativeModels.GenerateDataLM_SkewedPrior import make_lm_program_gamma_prior_batched, make_lm_program_gamma_prior, make_lm_program_gamma_prior_intercept_batched, make_lm_program_gamma_prior_intercept
from PFNExperiments.LinearRegression.GenerativeModels.GenerateData_LogReg import make_logreg_program_ig_batched, make_logreg_program_ig, make_logreg_program_ig_intercept_batched, make_logreg_program_ig_intercept
from PFNExperiments.LatentFactorModels.GenerativeModels.Clustering.GMMs import make_gmm_program_univariate_batched, make_gmm_program_univariate
from PFNExperiments.LatentFactorModels.GenerativeModels.Clustering.GMMs import make_gmm_program_spherical_batched, make_gmm_program_spherical
from PFNExperiments.LatentFactorModels.GenerativeModels.Clustering.GMMs import make_gmm_program_diagonal_batched, make_gmm_program_diagonal

"""
This file provides the mapping from the name of the program to the probabilistic program.
"""

name2pprogram_maker = {
    "ig": (make_lm_program_ig_batched, make_lm_program_ig),
    "ig_intercept": (make_lm_program_ig_intercept_batched, make_lm_program_ig_intercept),
    "Laplace_ig": (make_lm_program_Laplace_ig_batched, make_lm_program_Laplace_ig),
    "Laplace_ig_intercept": (make_lm_program_Laplace_ig_intercept_batched, make_lm_program_Laplace_ig_intercept),
    "Gamma_ig": (make_lm_program_gamma_prior_batched, make_lm_program_gamma_prior),
    "Gamma_ig_intercept": (make_lm_program_gamma_prior_intercept_batched, make_lm_program_gamma_prior_intercept),
    "logreg_ig": (make_logreg_program_ig_batched, make_logreg_program_ig),
    "logreg_ig_intercept": (make_logreg_program_ig_intercept_batched, make_logreg_program_ig_intercept),
    "gmm_univariate": (make_gmm_program_univariate_batched, make_gmm_program_univariate),
    "gmm_spherical": (make_gmm_program_spherical_batched, make_gmm_program_spherical),+
    "gmm_diagoanl": (make_gmm_program_diagonal_batched, make_gmm_program_diagonal)
}