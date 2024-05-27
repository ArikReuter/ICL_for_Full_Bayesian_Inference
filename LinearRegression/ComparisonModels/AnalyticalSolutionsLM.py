import torch 
from PFNExperiments.LinearRegression.GenerativeModels.LM_abstract import ppgram_linear_model_return_y
from PFNExperiments.LinearRegression.ComparisonModels.PosteriorComparisonModel import PosteriorComparisonModel
from PFNExperiments.LinearRegression.GenerativeModels.GenerateDataLM_Examples import make_lm_program_ig_batched
from pyro.distributions.multivariate_studentt import MultivariateStudentT

class PosteriorLM_IG(PosteriorComparisonModel):
    """
    Class that defines the analytical solution for the posterior of a linear model with an inverse gamma prior
    """
    
    def __init__(self,
                 pprogram_dict = {
                    "tau": 1.0,
                    "a": 5.0,
                    "b": 2.0
                 },
                 n_samples:int = 10_000,
                 ) -> None:
        """
        Args:
            pprogram_dict: dict: the dictionary containing the parameters of the model
            n_samples: int: the number of samples to draw from the posterior
        """
        
        self.pprogram_dict = pprogram_dict
        self.n_samples = n_samples

    def sample_posterior(self,
                            X: torch.Tensor,
                            y: torch.Tensor) -> torch.Tensor:
            """
            A method that samples from the posterior distribution
            Args:
                X: torch.Tensor: the covariates, of shape (batch_size, n, p)
                y: torch.Tensor: the response variable of shape (batch_size, n)
            Returns:
                torch.Tensor: the samples from the posterior distribution
            """

            if len(X.shape) == 2:
                X = X.unsqueeze(0)
                y = y.unsqueeze(0)



            assert X.shape[0] == y.shape[0], "The number of batches in X and y must be the same"
            assert X.shape[1] == y.shape[1], "The number of samples in X and y must be the same"

            tau = self.pprogram_dict["tau"]
            a = self.pprogram_dict["a"]
            b = self.pprogram_dict["b"]
            
            bs, n, p = X.shape

            sigma_squared = tau**2 
            SIGMA = torch.eye(p) * sigma_squared

            XTX = torch.matmul(X.permute(0, 2, 1), X)

            SIGMA_post_inverse = torch.inverse(SIGMA) + XTX
            SIGMA_post = torch.inverse(SIGMA_post_inverse)
            
            XTy = torch.matmul(X.permute(0, 2, 1), y.unsqueeze(-1))
            w_post = torch.matmul(SIGMA_post, XTy)

            a_post = a + n/2

            b_post = b + 0.5 * (torch.linalg.norm(y, dim = 1)**2 - (w_post.permute(0, 2, 1) @ SIGMA_post_inverse @ w_post).squeeze()).squeeze(-1)

            full_cov_posterior_w = SIGMA_post * (b_post/a_post).unsqueeze(-1).unsqueeze(-1)

            tril_matrix = torch.linalg.cholesky(full_cov_posterior_w)


            a_post = torch.tensor(a_post).expand(bs)

            w_post = w_post.squeeze()
            
            posterior_dist_w = MultivariateStudentT(loc=w_post, scale_tril=tril_matrix, df=2*a_post)


            assert torch.linalg.norm(posterior_dist_w.covariance_matrix - full_cov_posterior_w) < 1e-5, "The covariance matrices are not the same. The difference is: {}".format(torch.linalg.norm(posterior_dist_w.covariance_matrix - full_cov_posterior_w))

            samples = posterior_dist_w.sample((self.n_samples,)).squeeze()

            result = {}
            result["beta"] = samples

            return result
    




