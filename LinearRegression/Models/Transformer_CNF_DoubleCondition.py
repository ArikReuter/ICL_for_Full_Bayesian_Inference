import torch 
import torch.nn as nn
import torch.nn.functional as F


import torch 
import torch.nn as nn
import torch.nn.functional as F
import math

from PFNExperiments.LinearRegression.Models.Transformer_CNF import Linear_block, Linear_skip_block, PositionwiseFeedForward, PositionalEncoding, TransformerEncoder, MLP, ConditionalLayerNorm, Rescale, EncoderBlockConditional

class ConditionalBatchNormDouble(nn.Module):
    def __init__(self, num_features_in_feat, num_features_in_cond_a, num_features_in_cond_b, num_features_out):
        super().__init__()
        self.num_features_in_feat = num_features_in_feat
        self.num_features_in_cond_a = num_features_in_cond_a
        self.num_features_in_cond_b = num_features_in_cond_b
        self.num_features_out = num_features_out

        self.bn = nn.BatchNorm1d(num_features_in_feat, affine=False)
        self.linear_gamma_a = nn.Linear(num_features_in_cond_a, num_features_out)
        self.linear_beta_a = nn.Linear(num_features_in_cond_a, num_features_out)
        self.linear_gamma_b = nn.Linear(num_features_in_cond_b, num_features_out)
        self.linear_beta_b = nn.Linear(num_features_in_cond_b, num_features_out)

    def forward(self, x, cond_a, cond_b):
        # Normalize the input
        x = self.bn(x)

        # Calculate the gamma and beta parameters
        gamma_a = self.linear_gamma_a(cond_a)
        beta_a = self.linear_beta_a(cond_a)
        gamma_b = self.linear_gamma_b(cond_b)
        beta_b = self.linear_beta_b(cond_b)

        # Apply the conditional scaling and shifting
        x = gamma_a * x + beta_a
        x = gamma_b * x + beta_b
        return x    


class MLPConditionalDouble(nn.Module):
    """"
    An MLP where after each skip layer a conditional batch norm layer is applied
    """

    def __init__(self, n_input_units, n_output_units, n_hidden_units, n_skip_layers, dropout_rate, n_condition_features_a, n_condition_features_b):

        super(MLPConditionalDouble, self).__init__()
        self.n_input_units = n_input_units
        self.n_hidden_units = n_hidden_units
        self.n_skip_layers = n_skip_layers
        self.dropout_rate = dropout_rate
        self.n_output_units = n_output_units
        self.n_condition_features_a = n_condition_features_a
        self.n_condition_features_b = n_condition_features_b

        self.linear1 = Linear_block(n_input_units, n_hidden_units, dropout_rate)    # initial linear layer
        self.conditional_bn1 = ConditionalBatchNormDouble(num_features_in_feat = n_hidden_units, num_features_in_cond_a = n_condition_features_a, num_features_in_cond_b = n_condition_features_b, num_features_out = n_hidden_units)

        self.hidden_bn_layers = torch.nn.ModuleList([ConditionalBatchNormDouble(n_hidden_units, n_condition_features_a, n_condition_features_b, n_hidden_units) for _ in range(n_skip_layers)])
        self.hidden_layers = torch.nn.ModuleList([Linear_skip_block(n_hidden_units, dropout_rate) for _ in range(n_skip_layers)])

        self.linear_final =  torch.nn.Linear(n_hidden_units, n_output_units)
        self.conditional_bn_final = ConditionalBatchNormDouble(n_output_units, n_condition_features_a, n_condition_features_b, n_output_units)

    def forward(self, x, condition_a, condition_b):
        x = self.linear1(x)
        x = self.conditional_bn1(x, condition_a, condition_b)
        
        for hidden_layer, hidden_bn_layer in zip(self.hidden_layers, self.hidden_bn_layers):
            x = hidden_layer(x)
            x = hidden_bn_layer(x, condition_a, condition_b)

        x = self.linear_final(x)
        x  = self.conditional_bn_final(x, condition_a, condition_b)

        return(x)


class ConditionalLayerNormDouble(nn.Module):
    """
    Conditional Layer Normalization with two conditions
    """
    def __init__(
            self,
            d_model: int, 
            n_condition_features_a: int, 
            n_condition_features_b: int
    ):
        """
        Args:
            d_model: int: the model dimension
            n_condition_features_a: int: the number of features in the first conditioning tensor
            n_condition_features_b: int: the number of features in the second conditioning tensor
        """
        super(ConditionalLayerNormDouble, self).__init__()
        self.conditional_layer_norm_a = ConditionalLayerNorm(d_model, n_condition_features_a)
        self.conditional_layer_norm_b = ConditionalLayerNorm(d_model, n_condition_features_b)

    def forward(self, x: torch.tensor, condition_a: torch.tensor, condition_b: torch.tensor) -> torch.tensor:
        """
        Applies conditional layer normalization to the input tensor x given the condition tensors.
        Args:
            x: torch.tensor: the input tensor of shape (n, seq_len, d_model)
            condition_a: torch.tensor: the first condition tensor of shape (n, n_condition_features_a)
            condition_b: torch.tensor: the second condition tensor of shape (n, n_condition_features_b)
        Returns:
            torch.tensor: the output tensor of shape (n, seq_len, d_model)
        """
        x = self.conditional_layer_norm_a(x, condition_a)
        x = self.conditional_layer_norm_b(x, condition_b)
        return x


   
class RescaleDouble(nn.Module):
    """
    A class that rescales each element in the input sequence by a learnable paramter in each dimension.
    """
    def __init__(self, d_model: int, n_condition_features_a: int, n_condition_features_b: int, initialize_with_zeros: bool = True):
          """
          Args:
                d_model: int: the model dimension
                n_condition_features_a: int: the number of features in the first conditioning tensor
                n_condition_features_b: int: the number of features in the second conditioning tensor
          """
          super(RescaleDouble, self).__init__()
          self.rescale_a = Rescale(d_model, n_condition_features_a, initialize_with_zeros)
          self.rescale_b = Rescale(d_model, n_condition_features_b, initialize_with_zeros)

    def forward(self, x: torch.tensor, condition_a: torch.tensor, condition_b: torch.tensor) -> torch.tensor:
        """
        Rescales the input tensor x by a learnable parameter in each dimension given the condition tensors.
        Args:
            x: torch.tensor: the input tensor of shape (n, seq_len, d_model)
            condition_a: torch.tensor: the first condition tensor of shape (n, n_condition_features_a)
            condition_b: torch.tensor: the second condition tensor of shape (n, n_condition_features_b)

        Returns:
            torch.tensor: the output tensor of shape (n, seq_len, d_model)
        """
        x = self.rescale_a(x, condition_a)
        x = self.rescale_b(x, condition_b)
        return x
       

   
class EncoderBlockConditionalDouble(nn.Module):
   """  
   A transformer encoder block where after the layer norm the output is conditioned on an input tensor.
   Also after the multihead attention layer and the position
   """

   def __init__(
    self,
    d_model: int,
    n_heads: int,
    d_ff: int,
    dropout: float,
    n_condition_features_a: int,
    n_condition_features_b: int
    ):
      """
      Args:
            d_model: int: the model dimension
            n_heads: int: the number of heads in the multihead attention
            d_ff: int: the hidden dimension of the position wise feed forward network
            dropout: float: the dropout rate
            n_condition_features_a: int: the number of features in the first conditioning tensor
            n_condition_features_b: int: the number of features in the second conditioning tensor
      """
      super(EncoderBlockConditional, self).__init__()

      self.condition_layer_norm0 = ConditionalLayerNormDouble(d_model, n_condition_features_a, n_condition_features_b)
      self.multihead_attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
      self.rescale0 = RescaleDouble(d_model, n_condition_features_a, n_condition_features_b)

      self.condition_layer_norm1 = ConditionalLayerNormDouble(d_model, n_condition_features_a, n_condition_features_b)
      self.positionwise_feedforward = PositionwiseFeedForward(d_model, d_ff, d_model, dropout)
      self.rescale1 = RescaleDouble(d_model, n_condition_features_a, n_condition_features_b)


   def forward(self, x: torch.tensor, condition_a: torch.tensor, condition_b: torch.tensor) -> torch.tensor:
        """
        Forward pass for the encoder block.
        Args:
            x: torch.tensor: the input tensor of shape (n, seq_len, d_model)
            condition_a: torch.tensor: the first condition tensor of shape (n, n_condition_features_a)
            condition_b: torch.tensor: the second condition tensor of shape (n, n_condition_features_b)

        Returns:
            torch.tensor: the output tensor of shape (n, seq_len, d_model)
        """
        x = self.condition_layer_norm0(x, condition_a, condition_b)  # adaptive layer norm

        x_att, _ = self.multihead_attention(x, x, x)  # multihead attention

        x_att = self.rescale0(x_att, condition_a, condition_b)  # rescale

        x = x + x_att # apply residual connection 

        x = self.condition_layer_norm1(x, condition_a, condition_b) # apply layer norm

        x_ff = self.positionwise_feedforward(x) # apply position wise feed forward network

        x_ff = self.rescale1(x_ff, condition_a, condition_b) # rescale

        x = x + x_ff

        # apply residual connection and layer norm
        
        return x
     

   
class DecoderBlockConditionalDouble(nn.Module):
    """  
    A transformer decoder block where after the layer norm the output is conditioned on an input tensor.
    Also after the multihead attention layer and the position
    """
    
    def __init__(
     self,
     d_model_decoder: int,
     d_model_encoder: int,
     n_heads: int,
     d_ff: int,
     dropout: float,
     n_condition_features_a: int,
     n_condition_features_b: int,
     use_self_attention: bool = True
     ):
        """
        Args:
                d_model: int: the model dimension
                n_heads: int: the number of heads in the multihead attention
                d_ff: int: the hidden dimension of the position wise feed forward network
                dropout: float: the dropout rate
                n_condition_features_a: int: the number of features in the first conditioning tensor
                n_condition_features_b: int: the number of features in the second conditioning tensor
                use_self_attention: bool: whether to use self attention
        """
        super(DecoderBlockConditionalDouble, self).__init__()
        
        self.use_self_attention = use_self_attention
        if use_self_attention:
          self.condition_layer_norm0 = ConditionalLayerNormDouble(d_model_decoder, n_condition_features_a, n_condition_features_b)
          self.multihead_attention = nn.MultiheadAttention(d_model_decoder, n_heads, dropout=dropout, batch_first=True)
          self.rescale0 = RescaleDouble(d_model_decoder, n_condition_features_a, n_condition_features_b)
    
        self.condition_layer_norm1 = ConditionalLayerNormDouble(d_model_decoder, n_condition_features_a, n_condition_features_b)
        self.multihead_cross_attention = nn.MultiheadAttention(
            embed_dim=d_model_decoder,
            num_heads=n_heads,
                dropout=dropout,
                batch_first=True,
                kdim=d_model_encoder,
                vdim=d_model_encoder
        )
        self.rescale_cross = RescaleDouble(d_model_decoder, n_condition_features_a, n_condition_features_b)
    
        self.condition_layer_norm2 = ConditionalLayerNormDouble(d_model_decoder, n_condition_features_a, n_condition_features_b)
        self.positionwise_feedforward = PositionwiseFeedForward(d_model_decoder, d_ff, d_model_decoder, dropout)
        self.rescale1 = RescaleDouble(d_model_decoder, n_condition_features_a, n_condition_features_b)
    
    
    def forward(self, x: torch.tensor, x_encoder: torch.tensor, condition_a: torch.tensor, condition_b: torch.tensor) -> torch.tensor:
        """
        Forward pass for the encoder block.
        Args:
            x: torch.tensor: the input tensor of shape (n, seq_len, d_model)
            x_encoder: torch.tensor: the output tensor of the encoder block of shape (n, seq_len, d_model)
            condition_a: torch.tensor: the first condition tensor of shape (n, n_condition_features_a)
            condition_b: torch.tensor: the second condition tensor of shape (n, n_condition_features_b)

        Returns:
            torch.tensor: the output tensor of shape (n, seq_len, d_model)
        """
        if self.use_self_attention:
            x = self.condition_layer_norm0(x, condition_a, condition_b)
            x_att, _ = self.multihead_attention(x, x, x)  # multihead attention
            x_att = self.rescale0(x_att, condition_a, condition_b)
            x = x + x_att # apply residual connection 

        x = self.condition_layer_norm1(x, condition_a, condition_b)
        x_cross_att, _ = self.multihead_cross_attention(x, x_encoder, x_encoder)
        x_cross_att = self.rescale_cross(x_cross_att, condition_a, condition_b)
        x = x + x_cross_att

        x = self.condition_layer_norm2(x, condition_a, condition_b)
        x_ff = self.positionwise_feedforward(x) # apply position wise feed forward network
        x_ff = self.rescale1(x_ff, condition_a, condition_b)
        x = x + x_ff

        # apply residual connection and layer norm

        return x
        

   
class TransformerDecoderConditionalDouble(nn.Module):
    """
    A transformer decoder where the conditional DecoderBlock is used.
    """
    
    def __init__(
     self,
     n_input_features: int,
     d_model_decoder: int = 256,
     d_model_encoder: int = 256,
     n_heads: int = 8,
     d_ff: int = 512,
     dropout: float = 0.1,
     n_condition_features_a: int = 256,
     n_condition_features_b: int = 256,
     n_layers: int = 6,
     use_positional_encoding: bool = False,
     use_self_attention: bool = True
     ):
        """
        Args:
                d_model_decoder: int: the model dimension of the decoder
                d_model_encoder: int: the model dimension of the encoder
                n_heads: int: the number of heads in the multihead attention
                d_ff: int: the hidden dimension of the position wise feed forward network
                dropout: float: the dropout rate
                n_condition_features_a: int: the number of features in the first conditioning tensor
                n_condition_features_b: int: the number of features in the second conditioning tensor
                n_layers: int: the number of encoder blocks,
                use_positional_encoding: bool: whether to use positional encoding
                use_self_attention: bool: whether to use self attention
        """
        super(TransformerDecoderConditionalDouble, self).__init__()
    
        self.n_input_features = n_input_features
        self.d_model_decoder = d_model_decoder
        self.d_model_encoder = d_model_encoder
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.dropout = dropout
        self.n_condition_features_a = n_condition_features_a
        self.n_condition_features_b = n_condition_features_b
        self.n_layers = n_layers    
        self.use_positional_encoding = use_positional_encoding
        self.use_self_attention = use_self_attention
    
        self.embedding_layer = nn.Linear(n_input_features, d_model_decoder)
    
        if use_positional_encoding:
                self.positional_encoding = PositionalEncoding(d_model_decoder, dropout)
    
        self.decoder_blocks = nn.ModuleList([DecoderBlockConditionalDouble(d_model_decoder, d_model_encoder, n_heads, d_ff, dropout, n_condition_features_a, n_condition_features_b, use_self_attention=use_self_attention) for _ in range(n_layers)])
    
    
    def forward(self, x: torch.tensor, x_encoder: torch.tensor, condition_a: torch.tensor, condition_b: torch.tensor) -> torch.tensor:
            """
            Forward pass for the encoder.
            Args:
                    x: torch.tensor: the input tensor of shape (n, seq_len, d_model)
                    x_encoder: torch.tensor: the output tensor of the encoder block of shape (n, seq_len, d_model)
                    condition_a: torch.tensor: the first condition tensor of shape (n, n_condition_features_a)
                    condition_b: torch.tensor: the second condition tensor of shape (n, n_condition_features_b)
        
            Returns:
                    torch.tensor: the output tensor of shape (n, seq_len, d_model)
            """
            x = self.embedding_layer(x)
        
            if self.use_positional_encoding:
                    x = self.positional_encoding(x)
        
        
            for decoder_block in self.decoder_blocks:
                    x = decoder_block(x, x_encoder, condition_a, condition_b)
        
            return x
   

   
class TransformerConditionalDecoderDouble(nn.Module):
    """
    Uses an unconditional encoder and a conditional decoder
    """
    
    def __init__(
            self,
                n_input_features_encoder: int,
                n_input_features_decoder: int,
                d_model_encoder: int = 256,
                d_model_decoder: int = 256,
                n_heads_encoder: int = 8,
                n_heads_decoder: int = 8,
                d_ff_encoder: int = 512,
                d_ff_decoder: int = 512,
                dropout_encoder: float = 0.1,
                dropout_decoder: float = 0.1,
                n_conditional_input_features_a: int =  1,
                n_conditional_input_features_b: int =  1,
                n_condition_features_a: int = 256,
                n_condition_features_b: int = 256,
                n_layers_condition_embedding: int = 1,
                n_layers_encoder: int = 6,
                n_layers_decoder: int = 4,
                use_positional_encoding_encoder: bool = False,
                use_positional_encoding_decoder: bool = False,
                use_self_attention_decoder: bool = True,
                d_final_representation_transformer_encoder: int = 256,
                n_final_layers_representation_transformer_encoder: int = 3,
     ):
        """
        Args:
                d_model_encoder: int: the model dimension of the encoder
                d_model_decoder: int: the model dimension of the decoder
                n_heads_encoder: int: the number of heads in the multihead attention of the encoder
                n_heads_decoder: int: the number of heads in the multihead attention of the decoder
                d_ff_encoder: int: the hidden dimension of the position wise feed forward network of the encoder
                d_ff_decoder: int: the hidden dimension of the position wise feed forward network of the decoder
                dropout_encoder: float: the dropout rate of the encoder
                dropout_decoder: float: the dropout rate of the decoder
                n_conditional_input_features_a: int: the number of features in the first input tensor of the encoder
                n_conditional_input_features_b: int: the number of features in the second input tensor of the encoder
                n_condition_features_a: int: the number of features in the first conditioning tensor
                n_condition_features_b: int: the number of features in the second conditioning tensor
                n_layers_condition_embedding_encoder: int: the number of layers in the MLP that embeds the condition tensor of the encoder
                n_layers_encoder: int: the number of encoder blocks of the encoder
                n_layers_decoder: int: the number of decoder blocks of the decoder
                use_positional_encodign_encoder: bool: whether to use positional encoding in the encoder
                use_positional_encodign_decoder: bool: whether to use positional encoding in the decoder
                use_self_attention_decoder: bool: whether to use self attention in the decoder,
                d_final_representation_transformer_encoder: int: the hidden dimension of the final processing MLP of the encoder
                n_final_layers_representation_transformer_encoder: int: the number of layers in the final processing MLP of the encoder
        """
        super(TransformerConditionalDecoderDouble, self).__init__()
    
        self.n_conditional_input_features_a = n_conditional_input_features_a
        self.n_conditional_input_features_b = n_conditional_input_features_b
    
        self.condition_embedding_layer = MLP(
            n_input_units=n_conditional_input_features_a,
            n_output_units=n_condition_features_a,
            n_hidden_units=n_condition_features_a,
            n_skip_layers=n_layers_condition_embedding,
            dropout_rate=dropout_encoder
        )
    
    
    
        self.transformer_encoder = TransformerEncoder(
                n_input_features=n_input_features_encoder,
                    d_model=d_model_encoder,
                    n_heads=n_heads_encoder,
                    d_ff=d_ff_encoder,
                    dropout=dropout_encoder,
                    n_layers=n_layers_encoder,
                    use_positional_encoding=use_positional_encoding_encoder
                )

        self.MLP_representation_transformer_encoder = MLP(
            n_input_units=d_model_encoder,
            n_output_units=d_final_representation_transformer_encoder,
            n_hidden_units=d_final_representation_transformer_encoder,
            n_skip_layers=n_final_layers_representation_transformer_encoder,
            dropout_rate=dropout_encoder
        )
        
    
        self.transformer_decoder = TransformerDecoderConditionalDouble(
                n_input_features=n_input_features_decoder,
                    d_model_decoder=d_model_decoder,
                    d_model_encoder=d_model_encoder,
                    n_heads=n_heads_decoder,
                    d_ff=d_ff_decoder,
                    dropout=dropout_decoder,
                    n_condition_features_a=n_condition_features_a,
                    n_condition_features_b=n_condition_features_b,
                    n_layers=n_layers_decoder,
                    use_positional_encoding=use_positional_encoding_decoder,
                    use_self_attention=use_self_attention_decoder
        )

    def forward(self, x_encoder: torch.tensor, x_decoder: torch.tensor, condition_a: torch.tensor) -> torch.tensor:
            """
            Forward pass for the encoder.
            Args:
                    x_encoder: torch.tensor: the input tensor of shape (n, seq_len_encoder, d_model_encoder)
                    x_decoder: torch.tensor: the input tensor of shape (n, seq_len_decoder, d_model_decoder)
                    condition_a: torch.tensor: the first condition tensor of shape (n, n_condition_features_a)
        
            Returns:
                    torch.tensor: the output tensor of shape (n, seq_len_decoder, d_model_decoder)
                    torch.tensor: the output tensor of shape (n, n_condition_features_a)
                    torch.tensor: the output tensor of shape (n, n_condition_features_b)
            """
            condition_a = self.condition_embedding_layer(condition_a)

            x_encoder = self.transformer_encoder(x_encoder)  # has shape (n, seq_len_encoder, d_model_encoder)
            x_encoder_processed= torch.mean(x_encoder, dim=1)  # has shape (n, d_model_encoder)
            x_encoder_processed = self.MLP_representation_transformer_encoder(x_encoder_processed)  # has shape (n, d_final_representation_transformer_encoder)

            x_decoder = self.transformer_decoder(
                 x = x_decoder,
                 x_encoder = x_encoder,
                 condition_a = x_encoder_processed,
                 condition_b = x_encoder_processed
            )
        
            return x_decoder, condition_a, x_encoder_processed
    


class TransformerCNFConditionalDecoderDouble(TransformerConditionalDecoderDouble):
   """
   Use the TransformerConditional as a model for the CNF
   """

   def __init__(
         self,
         output_dim: int,
         d_final_processing: int = 256,
         n_final_layers: int = 3,
         dropout_final: float = 0.1,
        
         **kwargs
    ):
        """
        Args:
                output_dim: int: the output dimension
                d_final_processing: int: the hidden dimension of the final processing MLP
                n_final_layers: int: the number of layers in the final processing MLP
                dropout_final: float: the dropout rate of the final processing MLP
                d_final_representation_transformer_encoder: int: the hidden dimension of the final processing MLP of the transformer encoder
                treat_z_as_sequence: bool: whether to treat the latent variable z as a sequence. This transposes the latent variable z in the forward pass
        """

        super(TransformerCNFConditionalDecoderDouble, self).__init__(**kwargs)

        d_model_decoder = self.transformer_decoder.d_model_decoder

    
        self.final_processing = MLPConditionalDouble(
            n_input_units=d_model_decoder,
            n_output_units=output_dim,
            n_hidden_units=d_final_processing,
            n_skip_layers=n_final_layers,
            dropout_rate=dropout_final,
            n_condition_features_a=self.transformer_decoder.n_condition_features_a,
            n_condition_features_b=self.transformer_decoder.n_condition_features_b
        )


   def forward(self, z:torch.Tensor, x: torch.tensor, t: torch.tensor):
      """
      Args:
            z: torch.Tensor: the input latent variable of shape (BATCH_SIZE, 1, N_Latents)
            x: torch.Tensor: the data to condition on of shape (BATCH_SIZE, SEQ_LEN, N_INPUT_FEATURES)
            t: torch.Tensor: the time to condition on of shape (BATCH_SIZE, 1)
      """

      if not len(z.shape) == 3:
            z = z.unsqueeze(1)
     
      t = t.view(-1, 1)

      res_trafo, condition, x_encoder = super().forward(x, z, t)
      
    
      res_trafo = torch.mean(res_trafo, dim=1)


      res = self.final_processing(res_trafo, condition, x_encoder)

      return res