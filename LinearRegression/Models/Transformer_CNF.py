import torch 
import torch.nn as nn
import torch.nn.functional as F


import torch 
import torch.nn as nn
import torch.nn.functional as F
import math


class One_Layer_MLP(torch.nn.Module):
  def __init__(self,  n_input_units):

    super(One_Layer_MLP, self).__init__()

    self.n_input_units = n_input_units
    self.fc1 = torch.nn.Linear(n_input_units, 1)

  def forward(self, x):

    x = self.fc1(x)
    return x

class Linear_skip_block(nn.Module):
  """
  Block of linear layer + softplus + skip connection +  dropout  + batchnorm
  """
  def __init__(self, n_input, dropout_rate, affine = True):
    super(Linear_skip_block, self).__init__()

    self.fc = nn.Linear(n_input, n_input)
    self.act = torch.nn.LeakyReLU()

    self.bn = nn.BatchNorm1d(n_input, affine = affine)
    self.drop = nn.Dropout(dropout_rate)

  def forward(self, x):
    x0 = x
    x = self.fc(x)
    x = self.act(x)
    x = x0 + x
    x = self.drop(x)
    x = self.bn(x)

    return x

class Linear_block(nn.Module):
  """
  Block of linear layer dropout  + batchnorm
  """
  def __init__(self, n_input, n_output, dropout_rate, affine = True):
    super(Linear_block, self).__init__()

    self.fc = nn.Linear(n_input, n_output)
    self.act = torch.nn.LeakyReLU()
    self.bn = nn.BatchNorm1d(n_output, affine = affine)
    self.drop = nn.Dropout(dropout_rate)

  def forward(self, x):
    x = self.fc(x)
    x = self.act(x)
    x = self.drop(x)
    x = self.bn(x)

    return x

class MLP(nn.Module):
  def __init__(self, n_input_units, n_output_units, n_hidden_units, n_skip_layers, dropout_rate):

    super(MLP, self).__init__()
    self.n_input_units = n_input_units
    self.n_hidden_units = n_hidden_units
    self.n_skip_layers = n_skip_layers
    self.dropout_rate = dropout_rate
    self.n_output_units = n_output_units

    self.linear1 = Linear_block(n_input_units, n_hidden_units, dropout_rate)    # initial linear layer
    self.hidden_layers = torch.nn.Sequential(*[Linear_skip_block(n_hidden_units, dropout_rate) for _ in range(n_skip_layers)])  #hidden skip-layers

    self.linear_final =  torch.nn.Linear(n_hidden_units, n_output_units)

  def forward(self, x):
    x = self.linear1(x)
    x = self.hidden_layers(x)
    x = self.linear_final(x)

    return(x)
  
class ConditionalBatchNorm(nn.Module):
    def __init__(self, num_features_in_feat, num_features_in_cond, num_features_out):
        super().__init__()
        self.num_features_in_feat = num_features_in_feat
        self.num_features_in_cond = num_features_in_cond
        self.num_features_out = num_features_out

        self.bn = nn.BatchNorm1d(num_features_in_feat, affine=False)
        self.linear_gamma = nn.Linear(num_features_in_cond, num_features_out)
        self.linear_beta = nn.Linear(num_features_in_cond, num_features_out)

    def forward(self, x, cond):
        # Normalize the input
        x = self.bn(x)

        # Calculate the gamma and beta parameters
        gamma = self.linear_gamma(cond)
        beta = self.linear_beta(cond)

        # Apply the conditional scaling and shifting
        x = gamma * x + beta
        return x

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

class MLPConditional(nn.Module):
    """"
    An MLP where after each skip layer a conditional batch norm layer is applied
    """

    def __init__(self, n_input_units, n_output_units, n_hidden_units, n_skip_layers, dropout_rate, n_condition_features):

        super(MLPConditional, self).__init__()
        self.n_input_units = n_input_units
        self.n_hidden_units = n_hidden_units
        self.n_skip_layers = n_skip_layers
        self.dropout_rate = dropout_rate
        self.n_output_units = n_output_units
        self.n_condition_features = n_condition_features

        self.linear1 = Linear_block(n_input_units, n_hidden_units, dropout_rate)    # initial linear layer
        self.conditional_bn1 = ConditionalBatchNorm(n_hidden_units, n_condition_features, n_hidden_units)

        self.hidden_bn_layers = torch.nn.ModuleList([ConditionalBatchNorm(n_hidden_units, n_condition_features, n_hidden_units) for _ in range(n_skip_layers)])
        self.hidden_layers = torch.nn.ModuleList([Linear_skip_block(n_hidden_units, dropout_rate) for _ in range(n_skip_layers)])

        self.linear_final =  torch.nn.Linear(n_hidden_units, n_output_units)
        self.conditional_bn_final = ConditionalBatchNorm(n_output_units, n_condition_features, n_output_units)

    def forward(self, x, condition):
        x = self.linear1(x)
        x = self.conditional_bn1(x, condition)
        for i in range(self.n_skip_layers):
            x = self.hidden_layers[i](x)
            x = self.hidden_bn_layers[i](x, condition)

        x = self.linear_final(x)
        x = self.conditional_bn_final(x, condition)

        return(x)
    
class MLP_multi_head(nn.Module):

  def __init__(self, base_mlp, n_heads, n_output_units_per_head = [5,5]):
    """
    use the same base_mlp for n_heads, but with different output layers
    """
    super(MLP_multi_head, self).__init__()

    self.base_mlp = base_mlp
    self.n_heads = n_heads
    self.n_output_units_per_head = n_output_units_per_head

    self.heads = [torch.nn.Linear(base_mlp.n_output_units, n_output_units_per_head[i]) for i in range(n_heads)]

    self.heads = torch.nn.ModuleList(self.heads)

    self.act = torch.nn.LeakyReLU()

  def forward(self, x):
    x = self.base_mlp(x)
    x = self.act(x) # use activation function after the last hidden layer
    x = [head(x) for head in self.heads]

    return x
  
class PositionwiseFeedForward(nn.Module):
    "Implements a position-wise feed-forward network."
    def __init__(self, 
                 d_model_in:int, 
                 d_ff:int, 
                 d_model_out:int, 
                 dropout: float =0.1):
        """
        Args:
            d_model_in: int: the input dimension of the position wise feed forward network
            d_ff: int: the hidden dimension of the position wise feed forward network
            d_model_out: int: the output dimension of the position wise feed forward network
            dropout: float: the dropout rate
        """
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model_in, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x:torch.tensor) -> torch.tensor:
        """
        Takes an input tensor and returns the output tensor after applying the position wise feed forward network.
        The input tensor has shape (n, seq_len, d_model_in) and the output tensor has shape (n, seq_len, d_model_out).
        Args:
            x: torch.tensor: the input tensor of shape (n, seq_len, d_model_in)
        Returns:
            torch.tensor: the output tensor of shape (n, seq_len, d_model_out)
        """
        x = self.w_1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.w_2(x)
        return x
    
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        """
        Sinusoidal positional encoding for transformer models.
        Args:
            d_model: int: the model dimension
            dropout: float: the dropout rate
            max_len: int: the maximum length of the sequence
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``(batch_size, seq_len, features)``
        """
        x = x.permute(1, 0, 2)
        x = x + self.pe[:x.size(0)]

        x = x.permute(1, 0, 2)
        
        return self.dropout(x)

class ConditionalLayerNorm(nn.Module):
    """
    Conditional Layer Normalization
    """
    def __init__(self, d_model: int, n_condition_features: int):
        """
        Args:
            d_model: int: the model dimension
            n_condition_features: int: the number of features in the conditioning tensor
        """
        super(ConditionalLayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.gamma_layer = nn.Linear(n_condition_features, d_model)
        self.beta_layer = nn.Linear(n_condition_features, d_model)

    def forward(self, x: torch.tensor, condition: torch.tensor) -> torch.tensor:
        """
        Applies conditional layer normalization to the input tensor x given the condition tensor.
        Args:
            x: torch.tensor: the input tensor of shape (n, seq_len, d_model)
            condition: torch.tensor: the condition tensor of shape (n, n_condition_features)
        Returns:
            torch.tensor: the output tensor of shape (n, seq_len, d_model)
        """
   
        gamma, beta = self.gamma_layer(condition), self.beta_layer(condition)

        # apply layer norm
        x = self.layer_norm(x)
        x = gamma.unsqueeze(1) * x + beta.unsqueeze(1)
        return x
    
class Rescale(nn.Module):
   """
   A class that rescales each element in the input sequence by a learnable paramter in each dimension.
   """
   def __init__(self, d_model: int, n_condition_features: int, initialize_with_zeros: bool = True):
        """
        Args:
            d_model: int: the model dimension
            n_condition_features: int: the number of features in the conditioning tensor
        """
        super(Rescale, self).__init__()
        self.rescale = nn.Linear(n_condition_features, d_model)

        if initialize_with_zeros:
            nn.init.zeros_(self.rescale.weight)
            nn.init.zeros_(self.rescale.bias)

   def forward(self, x: torch.tensor, condition: torch.tensor) -> torch.tensor:
        """
        Rescales the input tensor x by a learnable parameter in each dimension given the condition tensor.
        Args:
            x: torch.tensor: the input tensor of shape (n, seq_len, d_model)
            condition: torch.tensor: the condition tensor of shape (n, n_condition_features)
        Returns:
            torch.tensor: the output tensor of shape (n, seq_len, d_model)
        """
        rescale = self.rescale(condition)
        x = x * rescale.unsqueeze(1)
        return x
      
class EncoderBlock(nn.Module):
   """
   A plain transformer encoder block
   """
   
   def __init__(
    self,
    d_model: int,
    n_heads: int,
    d_ff: int,
    dropout: float
    ):
      """
      Args:
            d_model: int: the model dimension
            n_heads: int: the number of heads in the multihead attention
            d_ff: int: the hidden dimension of the position wise feed forward network
            dropout: float: the dropout rate
      """
      super(EncoderBlock, self).__init__()

      self.layer_norm0 = nn.LayerNorm(d_model)
      self.multihead_attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

      self.layer_norm1 = nn.LayerNorm(d_model)
      self.positionwise_feedforward = PositionwiseFeedForward(d_model, d_ff, d_model, dropout)


   def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Forward pass for the encoder block.
        Args:
            x: torch.tensor: the input tensor of shape (n, seq_len, d_model)
        Returns:
            torch.tensor: the output tensor of shape (n, seq_len, d_model)
        """
        x = self.layer_norm0(x) # apply layer norm
        x_att, _ = self.multihead_attention(x, x, x)  # multihead attention
        x = x + x_att # apply residual connection 

        x = self.layer_norm1(x) # apply layer norm
        x_ff = self.positionwise_feedforward(x) # apply position wise feed forward network
        x = x + x_ff

        # apply residual connection and layer norm
        
        return x
   
class TransformerEncoder(nn.Module):
    """
    A transformer encoder
    """
    
    def __init__(
     self,
     n_input_features: int,
     d_model: int = 256,
     n_heads: int = 8,
     d_ff: int = 512,
     dropout: float = 0.1,
     n_layers: int = 6,
     use_positional_encoding: bool = False
     ):
        """
        Args:
                d_model: int: the model dimension
                n_heads: int: the number of heads in the multihead attention
                d_ff: int: the hidden dimension of the position wise feed forward network
                dropout: float: the dropout rate
                n_layers: int: the number of encoder blocks,
                use_positional_encoding: bool: whether to use positional encoding
        """
        super(TransformerEncoder, self).__init__()
    
        self.n_input_features = n_input_features
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.dropout = dropout
        self.n_layers = n_layers    
        self.use_positional_encoding = use_positional_encoding
    
        self.embedding_layer = nn.Linear(n_input_features, d_model)
    
        if use_positional_encoding:
                self.positional_encoding = PositionalEncoding(d_model, dropout)
    
        self.encoder_blocks = nn.ModuleList([EncoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
    
    
    def forward(self, x: torch.tensor) -> torch.tensor:
          """
          Forward pass for the encoder.
          Args:
                x: torch.tensor: the input tensor of shape (n, seq_len, d_model)
          Returns:
                torch.tensor: the output tensor of shape (n, seq_len, d_model)
          """
          x = self.embedding_layer(x)
    
          if self.use_positional_encoding:
                x = self.positional_encoding(x)
    
    
          for encoder_block in self.encoder_blocks:
                x = encoder_block(x)
    
          return x

class EncoderBlockConditional(nn.Module):
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
    n_condition_features: int
    ):
      """
      Args:
            d_model: int: the model dimension
            n_heads: int: the number of heads in the multihead attention
            d_ff: int: the hidden dimension of the position wise feed forward network
            dropout: float: the dropout rate
            n_condition_features: int: the number of features in the conditioning tensor
      """
      super(EncoderBlockConditional, self).__init__()

      self.condition_layer_norm0 = ConditionalLayerNorm(d_model, n_condition_features)
      self.multihead_attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
      self.rescale0 = Rescale(d_model, n_condition_features)

      self.condition_layer_norm1 = ConditionalLayerNorm(d_model, n_condition_features)
      self.positionwise_feedforward = PositionwiseFeedForward(d_model, d_ff, d_model, dropout)
      self.rescale1 = Rescale(d_model, n_condition_features)


   def forward(self, x: torch.tensor, condition: torch.tensor) -> torch.tensor:
        """
        Forward pass for the encoder block.
        Args:
            x: torch.tensor: the input tensor of shape (n, seq_len, d_model)
            condition: torch.tensor: the condition tensor of shape (n, n_condition_features)
        Returns:
            torch.tensor: the output tensor of shape (n, seq_len, d_model)
        """
        x = self.condition_layer_norm0(x, condition) # adaptive layer norm

        x_att, _ = self.multihead_attention(x, x, x)  # multihead attention

        x_att = self.rescale0(x_att, condition) # rescale 

        x = x + x_att # apply residual connection 

        x = self.condition_layer_norm1(x, condition) # apply layer norm

        x_ff = self.positionwise_feedforward(x) # apply position wise feed forward network

        x_ff = self.rescale1(x_ff, condition) # rescale

        x = x + x_ff

        # apply residual connection and layer norm
        
        return x

class TransformerEncoderConditional(nn.Module):
   """
   A transformer encoder where the conditional EncoderBlock is used.
   """

   def __init__(
    self,
    n_input_features: int,
    d_model: int = 256,
    n_heads: int = 8,
    d_ff: int = 512,
    dropout: float = 0.1,
    n_condition_features: int = 256,
    n_layers: int = 6,
    use_positional_encoding: bool = False
    ):
      """
      Args:
            d_model: int: the model dimension
            n_heads: int: the number of heads in the multihead attention
            d_ff: int: the hidden dimension of the position wise feed forward network
            dropout: float: the dropout rate
            n_condition_features: int: the number of features in the conditioning tensor
            n_layers: int: the number of encoder blocks,
            use_positional_encoding: bool: whether to use positional encoding
      """
      super(TransformerEncoderConditional, self).__init__()

      self.n_input_features = n_input_features
      self.d_model = d_model
      self.n_heads = n_heads
      self.d_ff = d_ff
      self.dropout = dropout
      self.n_condition_features = n_condition_features
      self.n_layers = n_layers    
      self.use_positional_encoding = use_positional_encoding

      self.embedding_layer = nn.Linear(n_input_features, d_model)

      if use_positional_encoding:
            self.positional_encoding = PositionalEncoding(d_model, dropout)

      self.encoder_blocks = nn.ModuleList([EncoderBlockConditional(d_model, n_heads, d_ff, dropout, n_condition_features) for _ in range(n_layers)])


   def forward(self, x: torch.tensor, condition: torch.tensor) -> torch.tensor:
        """
        Forward pass for the encoder.
        Args:
            x: torch.tensor: the input tensor of shape (n, seq_len, d_model)
            condition: torch.tensor: the condition tensor of shape (n, n_condition_features)
        Returns:
            torch.tensor: the output tensor of shape (n, seq_len, d_model)
        """
        x = self.embedding_layer(x)

        if self.use_positional_encoding:
            x = self.positional_encoding(x)


        for encoder_block in self.encoder_blocks:
            x = encoder_block(x, condition)

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

class TransformerConditional(nn.Module):
   """
   Combines the TransformerEncoderConditional and TransformerDecoderConditional
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
            n_conditional_input_features: int =  1,
            n_condition_features: int = 256,
            n_layers_condition_embedding: int = 1,
            n_layers_encoder: int = 6,
            n_layers_decoder: int = 4,
            use_positional_encoding_encoder: bool = False,
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
      super(TransformerConditional, self).__init__()

      self.n_conditional_input_features = n_conditional_input_features

      self.condition_embedding_layer = MLP(n_conditional_input_features, n_condition_features, n_condition_features, n_layers_condition_embedding, dropout_encoder)



      self.transformer_encoder = TransformerEncoderConditional(
         n_input_features=n_input_features_encoder,
            d_model=d_model_encoder,
            n_heads=n_heads_encoder,
            d_ff=d_ff_encoder,
            dropout=dropout_encoder,
            n_condition_features=n_condition_features,
            n_layers=n_layers_encoder,
            use_positional_encoding=use_positional_encoding_encoder
        )
      

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

          x_encoder = self.transformer_encoder(x_encoder, condition)
          x_decoder = self.transformer_decoder(x_decoder, x_encoder, condition)
    
          return x_decoder, condition

class TransformerConditionalDecoder(nn.Module):
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
            n_conditional_input_features: int =  1,
            n_condition_features: int = 256,
            n_layers_condition_embedding: int = 1,
            n_layers_encoder: int = 6,
            n_layers_decoder: int = 4,
            use_positional_encoding_encoder: bool = False,
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
      super(TransformerConditionalDecoder, self).__init__()

      self.n_conditional_input_features = n_conditional_input_features

      self.condition_embedding_layer = MLP(n_conditional_input_features, n_condition_features, n_condition_features, n_layers_condition_embedding, dropout_encoder)



      self.transformer_encoder = TransformerEncoder(
            n_input_features=n_input_features_encoder,
                d_model=d_model_encoder,
                n_heads=n_heads_encoder,
                d_ff=d_ff_encoder,
                dropout=dropout_encoder,
                n_layers=n_layers_encoder,
                use_positional_encoding=use_positional_encoding_encoder
            )
      

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

          x_encoder = self.transformer_encoder(x_encoder)
          x_decoder = self.transformer_decoder(x_decoder, x_encoder, condition)
    
          return x_decoder, condition

class TransformerCNF(TransformerConditional):
   """
   Use the TransformerConditional as a model for the CNF
   """

   def __init__(
         self,
         output_dim: int,
         d_final_processing: int = 256,
         n_final_layers: int = 2,
         dropout_final: float = 0.1,
         treat_z_as_sequence: bool = False,
         **kwargs
    ):
        """
        Args:
                output_dim: int: the output dimension
                d_final_processing: int: the hidden dimension of the final processing MLP
                n_final_layers: int: the number of layers in the final processing MLP
                dropout_final: float: the dropout rate of the final processing MLP
                treat_z_as_sequence: bool = False, whether to treat the latent variable z as a sequence. This transposes the latent variable z in the forward pass
        """
        super(TransformerCNF, self).__init__(**kwargs)

        d_model_decoder = self.transformer_decoder.d_model_decoder
    
        self.final_processing = MLPConditional(
            n_input_units=d_model_decoder,
            n_output_units=output_dim,
            n_hidden_units=d_final_processing,
            n_skip_layers=n_final_layers,
            dropout_rate=dropout_final,
            n_condition_features=self.transformer_decoder.n_condition_features
        )

        self.treat_z_as_sequence = treat_z_as_sequence

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

      res_trafo, condition = super().forward(x, z, t)

      if not self.treat_z_as_sequence:
            res_trafo = res_trafo.squeeze(1)
      else:
          res_trafo = torch.mean(res_trafo, dim=1)

      res = self.final_processing(res_trafo, condition)

      return res

class TransformerCNFConditionalDecoder(TransformerConditionalDecoder):
   """
   Use the TransformerConditional as a model for the CNF
   """

   def __init__(
         self,
         output_dim: int,
         d_final_processing: int = 256,
         n_final_layers: int = 1,
         dropout_final: float = 0.1,
         treat_z_as_sequence: bool = False,
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
        super(TransformerCNFConditionalDecoder, self).__init__(**kwargs)

        d_model_decoder = self.transformer_decoder.d_model_decoder
        self.treat_z_as_sequence = treat_z_as_sequence
    
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

      res_trafo, condition = super().forward(x, z, t)
      
      if not self.treat_z_as_sequence:
            res_trafo = res_trafo.squeeze(1)
      else:
          res_trafo = torch.mean(res_trafo, dim=1)


      res = self.final_processing(res_trafo, condition)

      return res

