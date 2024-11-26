import torch 
import torch.nn as nn
import torch.nn.functional as F


import torch 
import torch.nn as nn
import torch.nn.functional as F
import math

from PFNExperiments.LinearRegression.Models.Transformer_CNF import One_Layer_MLP, Linear_skip_block, Linear_block, MLP, ConditionalBatchNorm, ConditionalBatchNormDouble, MLPConditional, MLP_multi_head, PositionwiseFeedForward, ConditionalLayerNorm, PositionalEncoding, Rescale


import torch
import torch.nn as nn
import torch.nn.functional as F

class SkipMLPBlock(nn.Module):
    """
    A plain MLP block with skip connections and batch normalization
    """
    
    def __init__(
        self,
        d_model: int,
        dropout: float,
        use_bn: bool = True,
        use_skip_connection: bool = True
    ):
        """
        Args:
            d_model: int: the model dimension
            dropout: float: the dropout rate
            use_bn: bool: whether to use batch normalization
            use_skip_connection: bool: whether to use skip connections
        """
        super(SkipMLPBlock, self).__init__()
        
        self.use_bn = use_bn
        self.use_skip_connection = use_skip_connection
        
        # Linear layer
        self.fc = nn.Linear(d_model, d_model)
        
        # Batch normalization
        if use_bn:
            self.bn = nn.BatchNorm1d(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Forward pass
        Args:
            x: torch.tensor: the input tensor of shape (batch_size, d_model)
        Returns:
            torch.tensor: the output tensor of shape (batch_size, d_model)
        """
        # Apply linear transformation
        out = self.fc(x)
        
        # Apply batch normalization if enabled
        if self.use_bn:
            out = self.bn(out)
        
        # Apply activation function
        out = F.relu(out)
        
        # Apply dropout
        out = self.dropout(out)
        
        # Add skip connection if enabled
        if self.use_skip_connection:
            out = out + x
        
        return out

   

class SkipMLPEncoder(nn.Module):
    """
    An encoder that embeds the input, flattens the sequence dimension, and processes
    it with stacked SkipMLPBlocks.
    """
    
    def __init__(
        self,
        n_input_features: int,
        seq_len: int,
        d_model: int = 256,
        n_layers: int = 6,
        dropout: float = 0.1,
        use_bn: bool = True,
        use_skip_connection: bool = True
    ):
        """
        Args:
            n_input_features: int: the number of input features
            seq_len: int: the sequence length
            d_model: int: the model dimension
            n_layers: int: the number of SkipMLP blocks
            dropout: float: the dropout rate
            use_bn: bool: whether to use batch normalization in SkipMLP blocks
            use_skip_connection: bool: whether to use skip connections in SkipMLP blocks
        """
        super(SkipMLPEncoder, self).__init__()
        
        self.seq_len = seq_len
        self.d_model = d_model
        
        # Embedding layer to transform input to the model dimension
        self.embedding_layer = nn.Linear(n_input_features, d_model)
        
        # Stacked SkipMLP blocks
        self.mlp_blocks = nn.ModuleList([
            SkipMLPBlock(d_model * seq_len, dropout, use_bn, use_skip_connection)
            for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        Args:
            x: torch.Tensor: input tensor of shape (batch_size, seq_len, n_input_features)
        Returns:
            torch.Tensor: output tensor of shape (batch_size, d_model)
        """
        # Apply embedding layer
        x = self.embedding_layer(x)  # Shape: (batch_size, seq_len, d_model)
        
        # Flatten sequence dimension
        batch_size = x.size(0)
        x = x.view(batch_size, -1)  # Shape: (batch_size, seq_len * d_model)
        
        # Pass through SkipMLP blocks
        for mlp_block in self.mlp_blocks:
            x = mlp_block(x)
        

        return x


class DecoderBlockConditional(nn.Module):
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
    n_condition_features: int,
    use_self_attention: bool = True
    ):
      """
      Args:
            d_model: int: the model dimension
            n_heads: int: the number of heads in the multihead attention
            d_ff: int: the hidden dimension of the position wise feed forward network
            dropout: float: the dropout rate
            n_condition_features: int: the number of features in the conditioning tensor
            use_self_attention: bool: whether to use self attention
      """
      super(DecoderBlockConditional, self).__init__()
      
      self.use_self_attention = use_self_attention
      if use_self_attention:
        self.condition_layer_norm0 = ConditionalLayerNorm(d_model_decoder, n_condition_features)
        self.multihead_attention = nn.MultiheadAttention(d_model_decoder, n_heads, dropout=dropout, batch_first=True)
        self.rescale0 = Rescale(d_model_decoder, n_condition_features)

      self.condition_layer_norm1 = ConditionalLayerNorm(d_model_decoder, n_condition_features)
      self.multihead_cross_attention = nn.MultiheadAttention(
         embed_dim=d_model_decoder,
         num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
            kdim=d_model_encoder,
            vdim=d_model_encoder
      )
      self.rescale_cross = Rescale(d_model_decoder, n_condition_features)

      self.condition_layer_norm2 = ConditionalLayerNorm(d_model_decoder, n_condition_features)
      self.positionwise_feedforward = PositionwiseFeedForward(d_model_decoder, d_ff, d_model_decoder, dropout)
      self.rescale1 = Rescale(d_model_decoder, n_condition_features)


   def forward(self, x: torch.tensor, x_encoder: torch.tensor, condition: torch.tensor) -> torch.tensor:
        """
        Forward pass for the encoder block.
        Args:
            x: torch.tensor: the input tensor of shape (n, seq_len, d_model)
            x_encoder: torch.tensor: the output tensor of the encoder block of shape (n, seq_len, d_model)
            condition: torch.tensor: the condition tensor of shape (n, n_condition_features)
        Returns:
            torch.tensor: the output tensor of shape (n, seq_len, d_model)
        """
        if self.use_self_attention:
            x = self.condition_layer_norm0(x, condition) # adaptive layer norm
            x_att, _ = self.multihead_attention(x, x, x)  # multihead attention
            x_att = self.rescale0(x_att, condition) # rescale 
            x = x + x_att # apply residual connection 

        x = self.condition_layer_norm1(x, condition) # apply layer norm
        x_cross_att, _ = self.multihead_cross_attention(x, x_encoder, x_encoder)
        x_cross_att = self.rescale_cross(x_cross_att, condition)
        x = x + x_cross_att

        x = self.condition_layer_norm2(x, condition) # apply layer norm
        x_ff = self.positionwise_feedforward(x) # apply position wise feed forward network
        x_ff = self.rescale1(x_ff, condition) # rescale
        x = x + x_ff

        # apply residual connection and layer norm
        
        return x

class TransformerDecoderConditional(nn.Module):
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
    n_condition_features: int = 256,
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
            n_condition_features: int: the number of features in the conditioning tensor
            n_layers: int: the number of encoder blocks,
            use_positional_encoding: bool: whether to use positional encoding
            use_self_attention: bool: whether to use self attention
      """
      super(TransformerDecoderConditional, self).__init__()

      self.n_input_features = n_input_features
      self.d_model_decoder = d_model_decoder
      self.d_model_encoder = d_model_encoder
      self.n_heads = n_heads
      self.d_ff = d_ff
      self.dropout = dropout
      self.n_condition_features = n_condition_features
      self.n_layers = n_layers    
      self.use_positional_encoding = use_positional_encoding
      self.use_self_attention = use_self_attention

      self.embedding_layer = nn.Linear(n_input_features, d_model_decoder)

      if use_positional_encoding:
            self.positional_encoding = PositionalEncoding(d_model_decoder, dropout)

      self.decoder_blocks = nn.ModuleList([DecoderBlockConditional(d_model_decoder, d_model_encoder, n_heads, d_ff, dropout, n_condition_features, use_self_attention=use_self_attention) for _ in range(n_layers)])


   def forward(self, x: torch.tensor, x_encoder: torch.tensor, condition: torch.tensor) -> torch.tensor:
        """
        Forward pass for the encoder.
        Args:
            x: torch.tensor: the input tensor of shape (n, seq_len, d_model)
            x_encoder: torch.tensor: the output tensor of the encoder block of shape (n, seq_len, d_model)
            condition: torch.tensor: the condition tensor of shape (n, n_condition_features). Now, the condition tensor is assumed to be already embedded

        Returns:
            torch.tensor: the output tensor of shape (n, seq_len, d_model)
        """
        x = self.embedding_layer(x)

        if self.use_positional_encoding:
            x = self.positional_encoding(x)


        for decoder_block in self.decoder_blocks:
            x = decoder_block(x, x_encoder, condition)

        return x


class SkipMLPConditionalDecoder(nn.Module):
   """
   Uses an unconditional encoder and a conditional decoder
   """

   def __init__(
         self,
            n_input_features_encoder: int,
            n_input_features_decoder: int,
            d_model_encoder: int = 256,
            d_model_decoder: int = 256,
            n_heads_decoder: int = 8,
            d_ff_decoder: int = 512,
            dropout_encoder: float = 0.1,
            dropout_decoder: float = 0.1,
            n_conditional_input_features: int =  1,
            n_condition_features: int = 256,
            n_layers_condition_embedding: int = 1,
            n_layers_encoder: int = 6,
            n_layers_decoder: int = 4,
            use_positional_encoding_decoder: bool = False,
            use_self_attention_decoder: bool = True
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
            n_conditional_input_features_encoder: int: the number of features in the input tensor of the encoder
            n_condition_features: int: the number of features in the conditioning tensor
            n_layers_condition_embedding_encoder: int: the number of layers in the MLP that embeds the condition tensor of the encoder
            n_layers_encoder: int: the number of encoder blocks of the encoder
            n_layers_decoder: int: the number of decoder blocks of the decoder
            use_positional_encodign_encoder: bool: whether to use positional encoding in the encoder
            use_positional_encodign_decoder: bool: whether to use positional encoding in the decoder
            use_self_attention_decoder: bool: whether to use self attention in the decoder
      """
      super(SkipMLPConditionalDecoder, self).__init__()

      self.n_conditional_input_features = n_conditional_input_features

      self.condition_embedding_layer = MLP(n_conditional_input_features, n_condition_features, n_condition_features, n_layers_condition_embedding, dropout_encoder)

      self.encoder = SkipMLPEncoder(
          n_input_features=n_input_features_encoder, 
          d_model=d_model_encoder, 
          n_layers=n_layers_encoder, 
          dropout=dropout_encoder, 
          use_bn=True, 
          use_skip_connection=True)

      self.transformer_decoder = TransformerDecoderConditional(
            n_input_features=n_input_features_decoder,
                d_model_decoder=d_model_decoder,
                d_model_encoder=d_model_encoder,
                n_heads=n_heads_decoder,
                d_ff=d_ff_decoder,
                dropout=dropout_decoder,
                n_condition_features=n_condition_features,
                n_layers=n_layers_decoder,
                use_positional_encoding=use_positional_encoding_decoder,
                use_self_attention=use_self_attention_decoder
      )

   def forward(self, x_encoder: torch.tensor, x_decoder: torch.tensor, condition: torch.tensor) -> torch.tensor:
          """
          Forward pass for the encoder.
          Args:
                x_encoder: torch.tensor: the input tensor of shape (n, seq_len_encoder, d_model_encoder)
                x_decoder: torch.tensor: the input tensor of shape (n, seq_len_decoder, d_model_decoder)
                condition: torch.tensor: the condition tensor of shape (n, n_condition_features). Now, the condition tensor is assumed to be already embedded
    
          Returns:
                torch.tensor: the output tensor of shape (n, seq_len_decoder, d_model_decoder)
                torch.tensor: the output tensor of shape (n, n_condition_features)
          """
          condition = self.condition_embedding_layer(condition)

          x_encoder = self.encoder(x_encoder)

          x_encoder = x_encoder.unsqueeze(1)
          
          x_decoder = self.transformer_decoder(x_decoder, x_encoder, condition)
    
          return x_decoder, condition


class SkipMLPCNFConditionalDecoder(SkipMLPConditionalDecoder):
   """
   Use the SkipMLPCNFConditional as a model for the CNF
   """

   def __init__(
         self,
         output_dim: int,
         d_final_processing: int = 256,
         n_final_layers: int = 1,
         dropout_final: float = 0.1,
         treat_z_as_sequence: bool = False,
         average_decoder_output: bool = True,
         **kwargs
    ):
        """
        Args:
                output_dim: int: the output dimension
                d_final_processing: int: the hidden dimension of the final processing MLP
                n_final_layers: int: the number of layers in the final processing MLP
                dropout_final: float: the dropout rate of the final processing MLP
                treat_z_as_sequence: bool: whether to treat the latent variable z as a sequence. This transposes the latent variable z in the forward pass
        """
        super(SkipMLPCNFConditionalDecoder, self).__init__(**kwargs)

        d_model_decoder = self.transformer_decoder.d_model_decoder
        self.treat_z_as_sequence = treat_z_as_sequence
        self.average_decoder_output = average_decoder_output
    
        self.final_processing = MLPConditional(
            n_input_units=d_model_decoder,
            n_output_units=output_dim,
            n_hidden_units=d_final_processing,
            n_skip_layers=n_final_layers,
            dropout_rate=dropout_final,
            n_condition_features=self.transformer_decoder.n_condition_features
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

      if self.treat_z_as_sequence:
            z = z.permute(0, 2, 1)
      
      t = t.view(-1, 1)

      res1, condition = super().forward(x, z, t)
      
      if not self.treat_z_as_sequence:
            res1 = res1.squeeze(1)
      else:
          if self.average_decoder_output:
            res1 = torch.mean(res1, dim=1)

      res = self.final_processing(res1, condition)

      return res

