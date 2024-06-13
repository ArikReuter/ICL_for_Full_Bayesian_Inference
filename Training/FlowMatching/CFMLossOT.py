import torch 

from PFNExperiments.Training.FlowMatching.CFMLoss import CFMLoss


class CFMLossOT(CFMLoss):
    """
    Implementation of the conditional flow matching loss with the optimal transport probability paths
    """

    def psi_t_conditional_fun(
            self,
            z: torch.Tensor,
            z_1: torch.Tensor,
            t: float,
    ) -> torch.Tensor:
        """
        Compute the conditional flow \psi_t(\vz | \vz_1) with arguments \vz, \vz_1 and t
        Args:
            z: torch.Tensor: the input tensor sampled from the base distribution, has shape (batch_size, n_features)
            z_1: torch.Tensor: the tensor from the data distribution, has shape (batch_size, n_features)
            t: float: the time step, has shape (batch_size, 1)
        """
        assert z.shape == z_1.shape, f"z shape {z.shape} does not match z_1 shape {z_1.shape}"


        z_t = (1 - (1 - self.sigma_min)*t)*z + t*z_1

        #print(f"z_t in function psi_t_conditional_fun: {z_t} with shape {z_t.shape}")

        return z_t
    
    def conditional_target_vf_fun(
            self,
            z: torch.Tensor,
            z_1: torch.Tensor,
            t: float,
    ) -> torch.Tensor:
        """
        Compute the target conditional flow u_t(\vz| \vz_1) with arguments \vz, \vz_1 and t
        Args:
            z: torch.Tensor: the input tensor sampled from the base distribution 
            z_1: torch.Tensor: the tensor from the data distribution
            t: float: the time step
        """
        #print(f"t in function conditional_target_vf_fun: {t} with shape {t.shape}")
        assert z.shape == z_1.shape, f"z shape {z.shape} does not match z_1 shape {z_1.shape}"

        vft_z0_z1 = (z_1 - (1- self.sigma_min)*z)/(1 - (1 - self.sigma_min)*t)

        return vft_z0_z1
    
    def __init__(
            self,
            sigma_min: float = 1e-4,
    ):
        """
        Args:
            sigma_min: float: the minimum value of the sigma
        """
        super(CFMLossOT, self).__init__()
        self.sigma_min = sigma_min