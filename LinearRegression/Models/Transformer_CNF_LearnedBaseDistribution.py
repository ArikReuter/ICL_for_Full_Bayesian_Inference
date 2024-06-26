import torch 

from PFNExperiments.LinearRegression.Models.Transformer import Transformer
from PFNExperiments.LinearRegression.Models.Transformer_CNF_DoubleCondition2 import TransformerDecoderConditionalDouble_parallel, MLPConditionalDouble_parallel
from PFNExperiments.LinearRegression.Models.ModelPosterior import ModelPosterior


class TransformerReturnRepresentations(Transformer):
    """
    The same as the Transformer model, but returns the final representation of the transformer encoder
    """

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Args: 
            x: torch.tensor: the input tensor of shape (n_batch_size, seq_len, n_features). If self.transpose_input is True, the input tensor should have the shape (n_batch_size, n_features, seq_len)
        Returns:
            torch.tensor: the output tensor of shape (n_batch_size, n_outputs, n_output_units_per_head[i])
            torch.tensor: the output tensor of shape (n_batch_size, seq_len, n_features)
        """

        if self.transpose_input:
            x = x.transpose(1,2) # transpose the input tensor if necessary to have the shape (n_batch_size, n_features, seq_len)

        

        x = self.mlp1(x)

        if self.use_positional_encoding:
            x = self.positional_encoding(x) # use positional encoding if necessary
        else:
            x = self.act1(x)

        x = self.encoder(x)
        x_encoder = x
        x = self.mlp2(x)
        x = self.act2(x)

        x = x.view(x.shape[0], -1) # flatten the output of the transformer
        x = self.mlp3(x)
        x = self.act3(x)
        x = [head(x) for head in self.final_heads]

        return x, x_encoder


class TransformerCNFConditionalDecoderDouble_parallel_learnedBaseDistribution(torch.nn.Module):
    """
    A transformer model that uses a pre-trained encoder to provide samples from the base distribution
    Usage is indented as follows: 

    1. Forward pass of the encoder pred_encoder, x_encoder = model.forward_encoder(x)  # x has shape (n_batch_size, seq_len, n_features), x_encoder has shape (n_batch_size, seq_len, n_d_model_endoder), pred will e a tuple of predictions from the heads
    2. Get samples from the base distribution samples = model.get_base_distribution_samples(pred_encoder, n_samples)  # samples has shape (n_samples, n_batch_size, seq_len, n_features), set n_samples equal to the batch size
    3. Use the samples from the base distribution to get the samples from the probability path with the CNF loss class
    4. Forward pass of the decoder pred_decoder = model.forward_decoder(samples, x_encoder, condition_time)  # pred_decoder has shape (n_samples, n_batch_size, seq_len, n_features)
    """

    def __init__(self,
                 encoder: TransformerReturnRepresentations,
                 model_posterior: ModelPosterior,
                 mlp_to_process_encoder_output: torch.nn.Module,
                 mlp_to_process_time_conditioning: torch.nn.Module,
                 decoder: TransformerDecoderConditionalDouble_parallel,
                 mlp_to_process_decoder_output: MLPConditionalDouble_parallel,
                 freeze_encoder: bool = True
                 ):
        """
        Args:
            encoder: TransformerReturnRepresentations: the pre-trained encoder that gives samples for the base distribution and provides the conditioning on x
            model_posterior: ModelPosterior: the model that takes in the output of the decoder and allows to compute samples from the posterior
            mlp_to_process_encoder_output: torch.nn.Module: an MLP to process the output of the encoder before passing it to the decoder
            mlp_to_process_time_conditioning: torch.nn.Module: an MLP to process the time conditioning before passing it to the decoder
            decoder: TransformerDecoderConditionalDouble_parallel: the decoder that takes in samples from the base distribution and the processes conditioning on x from the encoder
            mlp_to_process_decoder_output: torch.nn.Module: an MLP to process the output of the decoder before passing it to the model_posterior
            freeze_encoder: bool: whether to freeze the encoder during training
        """
        super(TransformerCNFConditionalDecoderDouble_parallel_learnedBaseDistribution, self).__init__()
        self.encoder = encoder
        self.model_posterior = model_posterior
        self.mlp_to_process_encoder_output = mlp_to_process_encoder_output
        self.mlp_to_process_time_conditioning = mlp_to_process_time_conditioning
        self.decoder = decoder
        self.mlp_to_process_decoder_output = mlp_to_process_decoder_output
        self.freeze_encoder = freeze_encoder

    def forward_encoder(self, x: torch.tensor) -> torch.tensor:
        """
        Forward pass of the encoder
        Args:
            x: torch.tensor: the input tensor of shape (n_batch_size, seq_len, n_features). If self.transpose_input is True, the input tensor should have the shape (n_batch_size, n_features, seq_len)
        Returns:
            torch.tensor: the output tensor of shape (n_batch_size, n_outputs, n_output_units_per_head[i])
            torch.tensor: the output tensor of shape (n_batch_size, seq_len, n_features)
        """
        if self.freeze_encoder:
            with torch.no_grad():
                x_encoder = self.encoder(x)

        else:
            x_encoder = self.encoder(x)

        return x_encoder

    def get_base_distribution_samples(self, encoder_prediction: torch.tensor) -> torch.tensor:
        """
        Get samples from the base distribution
        Args:
            encoder_prediction: torch.tensor: the output of the encoder
            n_samples: int: the number of samples to generate
        Returns:
            torch.tensor: the samples from the base distribution
        """
        if self.freeze_encoder:
            with torch.no_grad():
                samples = self.model_posterior.pred2posterior_samples(encoder_prediction, n_samples=1)
        else:
            samples = self.model_posterior.sample_reparametrization(encoder_prediction, n_samples=1)

        samples = samples.squeeze(0)  # remove the first dimension of the samples
        return samples
    

    
    def forward_decoder(self, z: torch.tensor, x_encoder: torch.tensor, condition_time: torch.tensor) -> torch.tensor:
            """
            Forward pass for the encoder.
            Args:
                    z: torch.tensor: the input tensor of shape (n, seq_len_z, d_model_decoder)
                    x_encoder: torch.tensor: the output tensor of the encoder block of shape (n, seq_len_z, d_model_decoder)
                    condition_time: torch.tensor: the second condition tensor of shape (n, n_condition_features_b)
        
            Returns:
                    torch.tensor: the output tensor of shape (n, seq_len_z, d_model_decoder)
            """

            x_encoder_processed = torch.mean(x_encoder, dim=1)  # average the output of the encoder over the sequence length
            x_encoder_processed = self.mlp_to_process_encoder_output(x_encoder_processed) # process the output of the encoder with an MLP

            condition_time = self.mlp_to_process_time_conditioning(condition_time)
            
            if len(z.shape) == 2:
                z = z.unsqueeze(1)  # add sequence length dimension to the input tensor

            decoder_output = self.decoder(
                x = z,
                x_encoder = x_encoder,
                condition_a = x_encoder_processed,
                condition_b = condition_time
            )

            decoder_output = decoder_output.squeeze(1) # remove the second dimension of the output

            decoder_output = self.mlp_to_process_decoder_output(
                x = decoder_output,
                condition_a = x_encoder_processed,
                condition_b = condition_time
            )

            return decoder_output