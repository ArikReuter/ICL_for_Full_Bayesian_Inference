from PFNExperiments.Training.TrainerCurriculumCNF import TrainerCurriculumCNF
import torch

class TrainerCurriculumCNF_LatentFactor(TrainerCurriculumCNF):
    """
    Trainer class for Latent Factor Models
    """


    def batch_to_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Get the loss from a batch       
        Args:
            batch: dict[str, torch.Tensor]: the batch
        Returns:
            torch.Tensor: the loss
        """
        x = batch["x"]

        z_1 = batch["beta"]     # sample from the ground truth distribution
        z_0 = batch['base_sample_beta']  # sample from the base distribution
        t = batch["time"]  # sample the time
        t = t.unsqueeze(-1) # add a dimension to the time tensor to give it shape (batch_size, 1)

        t = t.float()

        if self.use_same_timestep_per_batch:
            t = t[0] * torch.ones_like(t)

        if self.coupling is not None:
            z_0 = self.coupling.couple(z_1, z_0)  # align the samples from the base distribution and the probability path using the coupling

        if not self.using_OTLossGaussianBase:
            zt = self.loss_function.psi_t_conditional_fun(z_0, z_1, t) # compute the sample from the probability path

        else:
            zt, z_0_b = self.loss_function.psi_t_conditional_fun(
            z_0_a = z_0,
            z_1 = z_1,
            t = t,
            z_0_b = None
        )
        
        vt_model = self.model(zt, x, t)  # compute the vector field prediction by the model

        if not self.using_OTLossGaussianBase:
            loss = self.loss_function(vector_field_prediction = vt_model, z_0 = z_0, z_1 = z_1, t = t)  # compute the loss by comparing the model prediction to the target vector field
        
        else:
            loss = self.loss_function(
                 vector_field_prediction = vt_model,
                 z_0_a = z_0,
                 z_0_b = z_0_b,
                 z_1 = z_1,
                 t = t
            )
        
        return loss