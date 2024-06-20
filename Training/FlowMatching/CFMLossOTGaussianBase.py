import torch 

from PFNExperiments.Training.FlowMatching.CFMLoss import CFMLoss


class CFMLossOTGaussianBase(torch.nn.Module):
    """
    Implementation of the conditional flow matching loss with the optimal transport probability paths and an arbitrary Gaussian base distribution
    """

    def psi_t_conditional_fun(
            self,
            z_0_a: torch.Tensor,
            z_1: torch.Tensor,
            t: float,
            z_0_b: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Compute the conditional flow \psi_t(\vz_0 | \vz_1) with arguments \vz, \vz_1 and t
        Args:
            z_0_a: torch.Tensor: the input tensor sampled from the base distribution N(\mu_base, \Sigma_base), has shape (batch_size, n_features)
            z_1: torch.Tensor: the tensor from the data distribution, has shape (batch_size, n_features)
            prec_factor: torch.Tensor: the precision factor of the base distribution, has shape (batch_size, n_features). Have that prec_fac @ prec_fac^T = precision matrix
            t: float: the time step, has shape (batch_size, 1)
            z_0_b: torch.Tensor: another input tensor sampled from a standard normal distribution, has shape (batch_size, n_features)

        Returns:
            z_t: torch.Tensor: the output tensor of the conditional flow, has shape (batch_size, n_features)
        """
        if z_0_b is None:
            z_0_b = torch.randn_like(z_0_a)

        assert z_0_a.shape == z_0_b.shape, f"z_0_a shape {z_0_a.shape} does not match z_0_b shape {z_0_b.shape}"
        assert z_1.shape == z_0_a.shape, f"z_1 shape {z_1.shape} does not match z_0_a shape {z_0_a.shape}"


        z_t = (1-t)*z_0_a + t * self.sigma_min * z_0_b + z_1


        if z_0_b is None:
            return z_t, z_0_b
        else:
            return z_t
    
    
    def __init__(
            self,
            sigma_min: float = 1e-4,
    ):
        """
        Args:
            sigma_min: float: the minimum value of the sigma
        """
        super(CFMLossOTGaussianBase, self).__init__()
        self.sigma_min = sigma_min


    def __call__(self,
                 vector_field_prediction: torch.Tensor,
                 z_0_a: torch.Tensor,
                 z_0_b: torch.Tensor,
                 z_1: torch.Tensor,
                 t: float,

    ) -> torch.Tensor:
        """
        Implement the computation of the conditional flow matching loss
        Args:
            vector_field_prediction: torch.Tensor: the prediction of the vector field based on \psi_t(\vz_0|\vz_1) computed by the model
            z_0_a: torch.Tensor: the input tensor sampled from the base distribution
            z_0_b: torch.Tensor: another input tensor sampled from a standard normal distribution
            z_1: torch.Tensor: the tensor from the data distribution
            t: float: the time step
        """

        target = self.sigma_min * z_0_b + z_1 - z_0_a

        loss = torch.mean((vector_field_prediction - target)**2)

        return loss

