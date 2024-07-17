from PFNExperiments.Training.TrainerCurriculumCNF import TrainerCurriculumCNF 
import torch


class TrainerCurriculumCNF_LearnedBaseDist_plus_nll(TrainerCurriculumCNF):

    def __init__(
        self,
        stop_gradients_from_FM_to_encoder: bool = False,
        **kwargs,
    ):
        """
        Args:
            stop_gradients_from_FM_to_encoder: bool: whether to stop the gradients from the flow matching to the encoder
            kwargs: dict: the keyword arguments for the TrainerCurriculumCNF
        """
        super().__init__(**kwargs)
        self.stop_gradients_from_FM_to_encoder = stop_gradients_from_FM_to_encoder



    def batch_to_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Get the loss from a batch       
        Args:
            batch: dict[str, torch.Tensor]: the batch
        Returns:
            torch.Tensor: the loss
        """
        x = batch["x"]
        y = batch["y"]

        X_y = torch.cat([x, y.unsqueeze(-1)], dim = -1) # concatenate the x and y values to one data tensor

        z_1 = batch["beta"]     # sample from the ground truth distribution
        z_0 = batch['base_sample_beta']  # sample from the base distribution
        t = batch["time"]  # sample the time
        t = t.unsqueeze(-1) # add a dimension to the time tensor to give it shape (batch_size, 1)

        t = t.float()

        if self.use_same_timestep_per_batch:
            t = t[0] * torch.ones_like(t)

    
        encoder_prediction, encoder_representation = self.model.forward_encoder(X_y)  # get the final output of the encoder and the intermediate transformer representation
        
        base_distribution_samples = self.model.get_base_distribution_samples(encoder_prediction)  # sample from the base distribution

        if self.coupling is not None:
            base_distribution_samples = self.coupling.couple(z_1, base_distribution_samples)  # align the samples from the base distribution and the probability path using the coupling

        z_t, z_0_b = self.loss_function.psi_t_conditional_fun(  # compute the conditional flow
            z_0_a = base_distribution_samples,
            z_1 = z_1,
            t = t,
            z_0_b = None,
        )

        if self.stop_gradients_from_FM_to_encoder:
            z_t = z_t.detach()
            z_0_b = z_0_b.detach()


        if self.stop_gradients_from_FM_to_encoder:
            encoder_representation_to_decoder = encoder_representation.detach()
        else:
            encoder_representation_to_decoder = encoder_representation

        vector_field_prediction = self.model.forward_decoder(  # compute the vector field prediction by the model
            z = z_t,
            x_encoder= encoder_representation_to_decoder,
            condition_time= t
        )

        loss = self.loss_function(
            pred_for_nll_loss = encoder_prediction,
            vector_field_prediction = vector_field_prediction,  # compute the loss
            z_0_a = base_distribution_samples,
            z_0_b = z_0_b,
            z_1 = z_1,
            t = t
        )
        return loss