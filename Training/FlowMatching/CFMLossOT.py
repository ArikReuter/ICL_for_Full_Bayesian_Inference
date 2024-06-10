import torch 

from PFNExperiments.Training.FlowMatching.CFMLoss import CFMLoss

class CFMLossOT(CFMLoss):
    """
    Implementation of the conditional flow matching loss with the optimal transport probability paths
    """

    def psi_t_conditional_fun(
            self,
            z_0: torch.Tensor,
            z_1: torch.Tensor,
            t: float,
    ) -> torch.Tensor:
        """
        Compute the conditional flow \psi_t(\vz | \vz_1) with arguments \vz, \vz_1 and t
        Args:
            z: torch.Tensor: the input tensor sampled from the base distribution 
            z_1: torch.Tensor: the tensor from the data distribution
            t: float: the time step
        """
        z_t = (1 - (1 - self.sigma_min)*t)*z_0 + t*z_1

        return z_t
    
    def conditional_target_vf_fun(
            self,
            z_0: torch.Tensor,
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
        vft_z0_z1 = (z_1 - (1- self.sigma_min)*z_0)/(1 - (1 - self.sigma_min)*t)

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