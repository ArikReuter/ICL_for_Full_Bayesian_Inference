import torch 
import torch.nn as nn
import torch.nn.functional as F


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
  def __init__(self, n_input, dropout_rate):
    super(Linear_skip_block, self).__init__()

    self.fc = nn.Linear(n_input, n_input)
    self.act = torch.nn.LeakyReLU()

    self.bn = nn.BatchNorm1d(n_input, affine = True)
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
  def __init__(self, n_input, n_output, dropout_rate):
    super(Linear_block, self).__init__()

    self.fc = nn.Linear(n_input, n_output)
    self.act = torch.nn.LeakyReLU()
    self.bn = nn.BatchNorm1d(n_output, affine = True)
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
                 dropout=0.1):
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
    
    



class Transformer(nn.Module):
  """
  A simple transformer encoder model that takes as input a sequence of features of shape (n_batch_size, seq_len, n_features) and returns a list of output tensors of shape (n_batch_size, n_output_units_per_head[i]) for i in range(n_outputs).
  """
  def __init__(self, 
               n_features: int = 5, 
               seq_len: int = 100, 
               d_model: int = 256, 
               dim_feedforward :int = 256, 
               dropout_rate: int = 0.1, 
               n_heads: int = 4, 
               n_layers: int = 3, 
               n_skip_layers_final_heads: int = 2,
               n_output_units_per_head = [5,5],
               projection_size_after_trafo = None):
    
    """
    A simple transformer encoder model that takes as input a sequence of features of shape (n_batch_size, seq_len, n_features) and returns a list of output tensors of shape (n_batch_size, n_output_units_per_head[i]) for i in range(n_outputs).

    Args: 
        n_features: int: the number of input features
        seq_len: int: the sequence length
        d_model: int: the model dimension of the transformer, i.e. the number of the latents used for attention mechanism
        dim_feedforward: int: the dimension of the feed forward network in the transformer
        dropout_rate: float: the dropout rate
        n_heads: int: the number of heads
        n_layers: int: the number of layers
        n_skip_layers_final_heads: int: the number of skip layers in the final heads
        n_outputs: int: the number of output heads
        n_output_units_per_head: list: the number of output units per head
        projection_size_after_trafo: int: the projection size after the transformer
    """
    
    super(Transformer, self).__init__()

    self.seq_len = seq_len

    self.d_model = d_model
    self.dim_feedforward = dim_feedforward
    self.n_features = n_features
    self.dropout_rate = dropout_rate
    self.n_heads = n_heads
    self.n_layers = n_layers
    self.n_skip_layers_final_heads = n_skip_layers_final_heads
    self.n_output_units_per_head = n_output_units_per_head

    if projection_size_after_trafo is None:
      self.projection_size_after_trafo = d_model//4
    
    self.projection_size_after_trafo = projection_size_after_trafo
    self.n_outputs = len(n_output_units_per_head)

    self.mlp1 = PositionwiseFeedForward(d_model_in = n_features, d_ff = (d_model + n_features)//2, d_model_out = d_model, dropout = dropout_rate) # use a position-wise feed forward network on the initial input to scale it to the model dimension
    self.act1 = torch.nn.LeakyReLU()

    self.encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward, dropout_rate, batch_first=True) # use a transformer encoder layer 
    self.encoder = nn.TransformerEncoder(self.encoder_layer, n_layers) # use a transformer encoder based on the transformer encoder layer

    self.mlp2 = PositionwiseFeedForward(d_model, (d_model + self.projection_size_after_trafo)//2, self.projection_size_after_trafo, dropout_rate) # use a position-wise feed forward network to scale the output of the transformer to the projection size
    self.act2 = torch.nn.LeakyReLU()

    self.mlp3 = nn.Linear(seq_len * self.projection_size_after_trafo, dim_feedforward)  # use a mlp to mingle together the outputs of the transformer
    self.act3 = torch.nn.LeakyReLU()

    self.final_heads = torch.nn.ModuleList(MLP(n_input_units = dim_feedforward, 
                                               n_output_units = n_output_units_per_head[i], 
                                               n_hidden_units = dim_feedforward, 
                                               n_skip_layers = self.n_skip_layers_final_heads, 
                                               dropout_rate = dropout_rate) for i in range(self.n_outputs))  # use an individual mlp for each head


  def forward(self, x):
    """
    Args: 
        x: torch.tensor: the input tensor of shape (n_batch_size, seq_len, n_features)
    Returns:
        list: a list of output tensors of shape (n_batch_size, n_output_units_per_head[i]) for i in range(n_outputs)
    """
    x = self.mlp1(x)
    x = self.act1(x)

    x = self.encoder(x)
    x = self.mlp2(x)
    x = self.act2(x)

    x = x.view(x.shape[0], -1) # flatten the output of the transformer
    x = self.mlp2(x)
    x = self.act3(x)
    x = [head(x) for head in self.final_heads]

    return x