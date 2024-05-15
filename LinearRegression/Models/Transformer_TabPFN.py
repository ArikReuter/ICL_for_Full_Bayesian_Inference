import torch
import tabpfn
import numpy as np

from tabpfn.utils import normalize_data, to_ranking_low_mem, remove_outliers
from tabpfn.utils import NOP, normalize_by_used_features_f
from tabpfn.scripts.transformer_prediction_interface import column_or_1d, check_classification_targets
from sklearn.preprocessing import LabelEncoder



from sklearn.preprocessing import PowerTransformer, QuantileTransformer, RobustScaler
import warnings

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
    
    


class TransformerTabPFN(torch.nn.Module):
    """
    A class that uses the TabPFN Classifier as a backbone for a Transformer for Bayesian Inference.
    """

    def __init__(self,
                 seq_len:int ,
                 n_outputs:int,
                 n_output_units_per_head: list,
                 dim_feedforward:int,
                 dropout_rate:float,
                 projection_size_after_trafo:int,
                 n_skip_layers_final_heads:int):
        
        """
        Args:
            seq_len: int: the length of the sequence
            n_outputs: int: the number of outputs
            n_output_units_per_head: list: the number of output units for each head
            dim_feedforward: int: the number of hidden units in the feed forward network
            dropout_rate: float: the dropout rate
            projection_size_after_trafo: int: the projection size after the transformer
            n_skip_layers_final_heads: int: the number of skip layers in the final heads
        """
        
        super(TransformerTabPFN, self).__init__()

        self.seq_len = seq_len
        self.n_outputs = n_outputs
        self.n_output_units_per_head = n_output_units_per_head
        self.dim_feedforward = dim_feedforward
        self.dropout_rate = dropout_rate
        self.projection_size_after_trafo = projection_size_after_trafo
        self.n_skip_layers_final_heads = n_skip_layers_final_heads


        self.tabpfnclf = tabpfn.TabPFNClassifier(device='cpu', N_ensemble_configurations=1)   # load the TabPFN Classifier
        self.pfnbackbone = self.tabpfnclf.model[2]


        self.mlp2 = PositionwiseFeedForward(512, (512 + self.projection_size_after_trafo)//2, self.projection_size_after_trafo, dropout_rate) # use a position-wise feed forward network to scale the output of the transformer to the projection size
        self.act2 = torch.nn.LeakyReLU()

        self.mlp3 = nn.Linear(seq_len * self.projection_size_after_trafo, dim_feedforward)  # use a mlp to mingle together the outputs of the transformer
        self.act3 = torch.nn.LeakyReLU()

        self.final_heads = torch.nn.ModuleList(MLP(n_input_units = dim_feedforward, 
                                                n_output_units = n_output_units_per_head[i], 
                                                n_hidden_units = dim_feedforward, 
                                                n_skip_layers = self.n_skip_layers_final_heads, 
                                                dropout_rate = dropout_rate) for i in range(self.n_outputs))  # use an individual mlp for each head


    def validate_targets(self, y):
        """
        Validate targets.
        Args:
            y: array-like of shape (n_samples,)
        Returns:
            y: array-like of shape (n_samples,)
        """
        y_ = column_or_1d(y, warn=True)
        check_classification_targets(y)
        cls, y = np.unique(y_, return_inverse=True)
        if len(cls) < 2:
            raise ValueError(
                "The number of classes has to be greater than one; got %d class"
                % len(cls)
            )

        self.classes_ = cls

        return np.asarray(y, dtype=np.float64, order="C")
    
    def preprocess_train_data(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
    )-> tuple[torch.Tensor, torch.Tensor]:
        """
        Preprocess the input 'training' data for the transformer model and do some checks
        Args:
            X: torch.Tensor, x-value input
            y: torch.Tensor, ys input
        Returns:
            tuple[torch.Tensor, torch.Tensor]: preprocessed x and y values
        """
        
        
        y = self.validate_targets(y)
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(y)

        self.X_ = X
        self.y_ = y

        if (X.shape[1] > self.tabpfnclf.max_num_features):
            if self.tabpfnclf.subsample_features:
                print('WARNING: The number of features for this classifier is restricted to ', self.tabpfnclf.max_num_features, ' and will be subsampled.')
            else:
                raise ValueError("The number of features for this classifier is restricted to ", self.tabpfnclf.max_num_features)
        if len(np.unique(y)) > self.tabpfnclf.max_num_classes:
            raise ValueError("The number of classes for this classifier is restricted to ", self.tabpfnclf.max_num_classes)
        if X.shape[0] > 1024:
            raise ValueError("⚠️ WARNING: TabPFN is not made for datasets with a trainingsize > 1024. Prediction might take a while, be less reliable. We advise not to run datasets > 10k samples, which might lead to your machine crashing (due to quadratic memory scaling of TabPFN). Please confirm you want to run by passing overwrite_warning=True to the fit function.")

        return X, y

    def preprocess_eval_data(
            self, 
            eval_xs: torch.Tensor,
            eval_ys: torch.Tensor,

    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Preprocess the input 'evaluation' data for the transformer model
        Args:
            eval_xs: torch.Tensor, x-value input for evaluation
            eval_ys: torch.Tensor, ys input
        Returns:
            tuple[torch.Tensor, torch.Tensor]: preprocessed x and y values
        """
        r = preprocess_eval_data(
            eval_xs=eval_xs,
            eval_ys=eval_ys,
            preprocess_transform=  'none' if self.tabpfnclf.no_preprocess_mode else 'mix' ,
            eval_position=0,   # fix later!!!
            max_features=self.tabpfnclf.max_num_features,
            normalize_with_test=False,
            normalize_to_ranking=False,
            normalize_with_sqrt=False,
            device=self.tabpfnclf.device,
            categorical_feats=[],
        )
        return r


    def preprocess_data(self, X: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Preprocess the input data for the transformer model
        Args:
            X: torch.Tensor, x-value input
            y: torch.Tensor, ys input
        Returns:
            tuple[torch.Tensor, torch.Tensor]: preprocessed x and y values
        """
        pre = self.preprocess_train_data(X, y)
        pre = self.preprocess_eval_data(*pre)

        return pre


    def preprocess_data_batch(self, X: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Preprocess the input data for the transformer model in a batch
        Args:
            X: torch.Tensor, x-value input
            y: torch.Tensor, ys input
        Returns:
            tuple[torch.Tensor, torch.Tensor]: preprocessed x and y values
        """
        for i in range(X.shape[0]):

            print(X[i].shape, y[i].shape)
            X[i], y[i] = self.preprocess_data(X[i], y[i])
        return X, y

    def forward(self, x: torch.tensor, max_features = 100):
        """
        Args: 
            x: torch.tensor: the input tensor of shape (n_batch_size, seq_len, n_features). The last of the n_features should be the target variable.
        """
        y = x[:, :, -1]
        x = x[:, :, :-1]

        # normalize each feature in x
        x = x - x.mean(dim=1, keepdim=True)
        x = x / (x.std(dim=1, keepdim=True) + 1e-6)

        eval_xs_ = torch.cat([x[..., 0:],x[..., :0]],dim=-1) # this shifts the features by one


        eval_xs_ = torch.cat(
            [eval_xs_,
                torch.zeros((eval_xs_.shape[0], eval_xs_.shape[1], max_features- eval_xs_.shape[2]))], -1)


        x = self.pfnbackbone.encoder(eval_xs_)

        
        y = y.float().unsqueeze(-1)
        y = self.pfnbackbone.y_encoder(y)

        single_eval_pos = len(x)

        train_x = x[:single_eval_pos] + y[:single_eval_pos]
        #src = torch.cat([global_src, style_src, train_x, x_src[single_eval_pos:]], 0)


        t = self.pfnbackbone.transformer_encoder(train_x)
        
        t = self.mlp2(t)
        t = self.act2(t)

        t = t.view(t.shape[0], -1) # flatten the output of the transformer

        t = self.mlp3(t)
        t = self.act3(t)
        t = [head(t) for head in self.final_heads]

        return t

@staticmethod
def preprocess_eval_data(
                    eval_xs: torch.Tensor,
                    eval_ys: torch.Tensor,
                    preprocess_transform: str,
                    eval_position: int,
                    max_features: int,
                    normalize_with_test:bool = False,
                    normalize_to_ranking:bool = False,
                    normalize_with_sqrt:bool = False,
                    device:torch.device = torch.device('cpu') if not torch.cuda.is_available() else torch.device('cuda'),
                    categorical_feats:list = []) -> tuple[torch.Tensor, torch.Tensor]:
    
    """
    Preprocess the input data for the transformer model
    Args:
        eval_xs: torch.Tensor, x-value input for evaluation
        eval_ys: torch.Tensor, ys input
        preprocess_transform: str, type of preprocessing to be applied. Options: 'none', 'power', 'quantile', 'robust', 'power_all', 'quantile_all', 'robust_all'
        eval_position: int, position where the evaluation data starts
        max_features: int, maximum number of features to be used
        normalize_with_test: bool, whether to normalize with test data
        normalize_to_ranking: bool, whether to normalize to ranking
        normalize_with_sqrt: bool, whether to normalize with sqrt
        device: str, device to be used
        categorical_feats: list, list of categorical features
    """

    if eval_xs.shape[1] > 1:
        raise Exception("Transforms only allow one batch dim - TODO")

    if eval_xs.shape[2] > max_features:
        eval_xs = eval_xs[:, :, sorted(np.random.choice(eval_xs.shape[2], max_features, replace=False))]

    if preprocess_transform != 'none':
        if preprocess_transform == 'power' or preprocess_transform == 'power_all':
            pt = PowerTransformer(standardize=True)
        elif preprocess_transform == 'quantile' or preprocess_transform == 'quantile_all':
            pt = QuantileTransformer(output_distribution='normal')
        elif preprocess_transform == 'robust' or preprocess_transform == 'robust_all':
            pt = RobustScaler(unit_variance=True)

    # eval_xs, eval_ys = normalize_data(eval_xs), normalize_data(eval_ys)
    eval_xs = normalize_data(eval_xs, normalize_positions=-1 if normalize_with_test else eval_position)

    # Removing empty features
    eval_xs = eval_xs[:, 0, :]
    sel = [len(torch.unique(eval_xs[0:eval_ys.shape[0], col])) > 1 for col in range(eval_xs.shape[1])]
    eval_xs = eval_xs[:, sel]

    warnings.simplefilter('error')
    if preprocess_transform != 'none':
        eval_xs = eval_xs.cpu().numpy()
        feats = set(range(eval_xs.shape[1])) if 'all' in preprocess_transform else set(
            range(eval_xs.shape[1])) - set(categorical_feats)
        for col in feats:
            try:
                pt.fit(eval_xs[0:eval_position, col:col + 1])
                trans = pt.transform(eval_xs[:, col:col + 1])
                # print(scipy.stats.spearmanr(trans[~np.isnan(eval_xs[:, col:col+1])], eval_xs[:, col:col+1][~np.isnan(eval_xs[:, col:col+1])]))
                eval_xs[:, col:col + 1] = trans
            except:
                pass
        eval_xs = torch.tensor(eval_xs).float()
    warnings.simplefilter('default')

    eval_xs = eval_xs.unsqueeze(1)

    # TODO: Caution there is information leakage when to_ranking is used, we should not use it
    eval_xs = remove_outliers(eval_xs, normalize_positions=-1 if normalize_with_test else eval_position) \
            if not normalize_to_ranking else normalize_data(to_ranking_low_mem(eval_xs))
    # Rescale X
    eval_xs = normalize_by_used_features_f(eval_xs, eval_xs.shape[-1], max_features,
                                            normalize_with_sqrt=normalize_with_sqrt)

    
    
    
    
    return eval_xs.to(device), eval_ys.to(device)