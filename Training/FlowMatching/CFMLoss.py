import torch 
from typing import Callable


class CFMLoss(torch.nn.Module):
    """
    An implementation of the conditional flow matching loss
    """

    def __init__(
            self,
    ):
        """
        Args:
            psi_t_conditional_fun: Callable: the function to compute the conditional flow \psi_t(\vz | \vz_1) with arguments \vz, \vz_1 and t
            conditional_target_vf_fun: Callable: the function to compute the target conditional flow u_t(\vz| \vz_1) with arguments \vz, \vz_1 and t
        """
        super(CFMLoss, self).__init__()
        
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
        raise NotImplementedError
    
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
        raise NotImplementedError

    def get_element_conditional_probability_path(
            self,
            z_0: torch.Tensor,
            z_1: torch.Tensor,
            t: float,
    ) -> torch.Tensor:
        """
        Compute what the element from the conditional probability path should be. I.e. compute \psi_t(\vz_0|\vz_1). This will be one of the inputs to the model
        Args:
            z_0: torch.Tensor: the input tensor sampled from the base distribution 
            z_1: torch.Tensor: the tensor from the data distribution
            t: float: the time step
        """

        assert 0 <= t <= 1, f"t should be between 0 and 1, got {t}"
        assert z_0.shape == z_1.shape, f"z shape {z.shape} does not match z_1 shape {z_1.shape}"

        z_t = self.psi_t_conditional_fun(z_0, z_1, t)

        return z_t

    
    def __call__(self,
                 vector_field_prediction: torch.Tensor,
                 z_0: torch.Tensor,
                 z_1: torch.Tensor,
                 t: float,

    ) -> torch.Tensor:
        """
        Implement the computation of the conditional flow matching loss
        Args:
            vector_field_prediction: torch.Tensor: the prediction of the vector field based on \psi_t(\vz_0|\vz_1) computed by the model
            z_0: torch.Tensor: the input tensor sampled from the base distribution 
            z_1: torch.Tensor: the tensor from the data distribution
            t: float: the time step
        """

        assert 0 <= t <= 1, f"t should be between 0 and 1, got {t}"
        assert z_0.shape == z_1.shape, f"z shape {z_0.shape} does not match z_1 shape {z_1.shape}"

        ground_truth_vector_field = self.conditional_target_vf_fun(z_0, z_1, t)

        assert ground_truth_vector_field.shape == z_0.shape, f"ground truth vector field shape {ground_truth_vector_field.shape} does not match z_0 shape {z_0.shape}"

        loss = torch.nn.MSELoss()(vector_field_prediction, ground_truth_vector_field)

        return loss

