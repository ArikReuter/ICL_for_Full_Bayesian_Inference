import torch 

from PFNExperiments.Training.FlowMatching.CFMLossOTGaussianBase import CFMLossOTGaussianBase
from PFNExperiments.LinearRegression.Models.ModelPosterior import ModelPosterior

class CFMLossOTGaussianBase_plus_NLL(CFMLossOTGaussianBase):
    """
    Implementation of the conditional flow matching loss with the optimal transport probability paths and an arbitrary Gaussian base distribution
    """

   
    def __init__(
            self,
            weight_nll: float,
            NLL_Loss_function: callable,
            sigma_min: float = 1e-4,
            return_sub_losses: bool = False
    ):
        """
        Args:  
            weight_nll: float: the weight of the negative log likelihood, has to be between 0 and 1
            nll_loss_function: callable: the negative log likelihood loss function
            sigma_min: float: the minimum value of the sigma
            return_sub_losses: bool: whether to return the sub losses
        """
        super(CFMLossOTGaussianBase, self).__init__()

        assert 0 <= weight_nll <= 1, f"weight_nll has to be between 0 and 1, but is {weight_nll}"

        self.weight_nll = weight_nll
        self.NLL_Loss_function = NLL_Loss_function
        self.sigma_min = sigma_min
        self.return_sub_losses = return_sub_losses


    def __call__(self,
                 pred_for_nll_loss : torch.Tensor,
                 vector_field_prediction: torch.Tensor,
                 z_0_a: torch.Tensor,
                 z_0_b: torch.Tensor,
                 z_1: torch.Tensor,
                 t: float,

    ) -> torch.Tensor:
        """
        Implement the computation of the conditional flow matching loss
        Args:
            pred_for_nll_loss: torch.Tensor: the prediction of the negative log likelihood computed by the model
            vector_field_prediction: torch.Tensor: the prediction of the vector field based on \psi_t(\vz_0|\vz_1) computed by the model
            z_0_a: torch.Tensor: the input tensor sampled from the base distribution
            z_0_b: torch.Tensor: another input tensor sampled from a standard normal distribution
            z_1: torch.Tensor: the tensor from the data distribution
            t: float: the time step
        """

        target = self.sigma_min * z_0_b + z_1 - z_0_a

        loss_flow_matching = torch.mean((vector_field_prediction - target)**2)

        loss_nll = self.NLL_Loss_function(pred_for_nll_loss, z_1)

        loss = (1-self.weight_nll) * loss_flow_matching + self.weight_nll * loss_nll

        if self.weight_nll == 0.0:
            loss = loss_flow_matching
        if self.weight_nll == 1.0:
            loss = loss_nll

        if self.return_sub_losses:
            return loss, loss_flow_matching, loss_nll

        return loss

