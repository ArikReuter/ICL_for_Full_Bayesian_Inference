from PFNExperiments.Evaluation.RealWorldEvaluation.Preprocess_univariate_GMM import Preprocessor_GMM_univariate
from PFNExperiments.Evaluation.RealWorldEvaluation.Preprocess_multivariate_GMM import Preprocessor_GMM_multivariate

name2preprocessor = {
    "gmm_univariate": Preprocessor_GMM_univariate,
    "gmm_multivariate": Preprocessor_GMM_multivariate
}
