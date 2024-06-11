import torch 
import torch.nn as nn
import torch.nn.functional as F
import math

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
  

class MLP_CNF(nn.Module):
    def __init__(self,
                n_data_inputs: int,
                n_parameter_inputs: int,
                hidden_size: int,

    ):
        
        super(MLP_CNF, self).__init__()
        self.n_data_inputs = n_data_inputs
        self.n_parameter_inputs = n_parameter_inputs
        self.hidden_size = hidden_size

        self.time_embedding_layer = nn.Linear(1, hidden_size)
        
        self.parameter_processing = MLP(n_parameter_inputs, hidden_size, hidden_size, 2, 0.1)
        self.data_processing = MLP(n_data_inputs, hidden_size, hidden_size, 2, 0.1)

        self.final_processing1 = Linear_skip_block(hidden_size, 0.1)
        self.final_processing2 = Linear_skip_block(hidden_size, 0.1)
        self.final_processing3 = Linear_skip_block(hidden_size, 0.1)
        self.final_processing4 = Linear_skip_block(hidden_size, 0.1)

        self.final_layer = nn.Linear(hidden_size, n_parameter_inputs)


    def forward(self,
                z: torch.Tensor,
                x: torch.Tensor,
                t: torch.Tensor,
                ) -> torch.Tensor:
      
        """
        Args: 
            x: torch.Tensor: the data to condition on 
            z: torch.Tensor: the parameters or latent variables of interest
            t: torch.Tensor: the time
        """

        # flatten x over the last dimension
        x = x.view(x.shape[0], -1)

        t = self.time_embedding_layer(t)
        z = self.parameter_processing(z)   
        x = self.data_processing(x)
        
        #t_sm = torch.softmax(t, dim = 1) # softmax over the data inputs to use in GLU units 
        #x_sm = torch.softmax(x, dim = 1)

        zx1 = z 
        zx1 = self.final_processing1(zx1)

        zx2 = zx1 + x
        zx2t = zx2 + t  
        zx2t = self.final_processing2(zx2t)

        #zx3 = zx2t + zx1
        #zx3t = zx3 * t_sm
        zx3t = self.final_processing3(zx2t)

        #zx4 = zx3t + zx2
        #zx4t = zx4 * t_sm
        zx4t = self.final_processing4(zx3t)

        out = self.final_layer(zx4t)

        return out





import torch
import torch.nn as nn
import torch.nn.functional as F

class ConditionalBatchNorm(nn.Module):
    def __init__(self, num_features, embedding_dim):
        super().__init__()
        self.num_features = num_features
        self.gamma = nn.Linear(embedding_dim, num_features)
        self.beta = nn.Linear(embedding_dim, num_features)
        self.bn = nn.BatchNorm1d(num_features, affine=False)

    def forward(self, x, y):
        gamma = self.gamma(y)
        beta = self.beta(y)
        x = self.bn(x)
        return gamma * x + beta

class MLP_CNF_BN(nn.Module):
    def __init__(self, n_data_inputs: int, n_parameter_inputs: int, layers: list):
        super(MLP_CNF_BN, self).__init__()
        
        self.time_embedding = nn.Linear(1, layers[0])  # Assuming the time embedding size is the size of the first layer

        # Create the layers of the MLP
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # First layer specifically from input size to the first hidden layer size
        self.layers.append(nn.Linear(n_parameter_inputs, layers[0]))
        self.batch_norms.append(ConditionalBatchNorm(layers[0], layers[0]))

        # Remaining layers
        for i in range(1, len(layers)):
            self.layers.append(nn.Linear(layers[i - 1], layers[i]))
            self.batch_norms.append(ConditionalBatchNorm(layers[i], layers[i]))

        # Output layer
        self.output_layer = nn.Linear(layers[-1], n_parameter_inputs)

    def forward(self, z: torch.Tensor, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # Time conditioning
        t_emb = F.relu(self.time_embedding(t.view(-1, 1)))  # Ensure t is the correct shape

        # Process through all hidden layers
        for layer, norm in zip(self.layers, self.batch_norms):
            z = F.relu(layer(z))
            z = norm(z, t_emb)

        # Output layer
        output = self.output_layer(z)
        return output
