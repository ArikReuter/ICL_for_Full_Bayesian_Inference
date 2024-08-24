import torch 
from sklearn.preprocessing import PowerTransformer
from scipy.stats import boxcox

def scale_features_01_total(X):
    """
    Scale the features between 0 and 1 using the min and max of the entire dataset
    """
    return 0.95*(X - X.min()) / (X.max() - X.min()) + 0.025  # scale between 0.025 and 0.975

def scale_features_01(X, eps = 1e-3):
    """
    Scale the features between 0 and 1 using the min and max of each feature
    Args: 
        X: torch.Tensor: the features of shape (N,P)
        eps: float: a small number to avoid division by zero
    """
    mins = torch.min(X, dim = 1, keepdim=True) # shpe (N,1)
    maxs = torch.max(X, dim = 1, keepdim=True) # shape (N,1)

    X_scaled = (X - mins.values) / (maxs.values - mins.values + eps)

    return X_scaled


def scale_features_01_power_transform(X):
    """
    Scale the features between 0 and 1 using the min and max of each feature. Apply the power-transform per feature before.
    Args: 
        X: torch.Tensor: the features of shape (N,P)
    """
    pt = PowerTransformer()
    X = pt.fit_transform(X)
    X = torch.tensor(X, dtype = torch.float)

    mins = torch.min(X, dim = 1, keepdim=True) # shpe (N,1)
    maxs = torch.max(X, dim = 1, keepdim=True) # shape (N,1)

    X_scaled = (X - mins.values) / (maxs.values - mins.values)

    return X_scaled


def make_target_scaler(mu: float = 0.0, var: float = 1.0, power_transform:bool = True) -> callable:
    """
    Make a target scaler that scales the target to have mean mu and variance var averages the target
    Args:
        mu: float: the mean of the target
        var: float: the variance of the target
        power_transform: bool: whether to apply a power transform to the target
    Returns:
        callable: a target scaler
    """

    def target_scaler(y: torch.Tensor) -> torch.Tensor:
        """
        Scale the target to have mean mu and variance var
        Args:
            y: torch.Tensor: the target
        Returns:
            torch.Tensor: the scaled target
        """
        assert y.dim() == 1, "The target should be 1D"
        if power_transform:
            pt = PowerTransformer()
            y = pt.fit_transform(y.reshape(-1,1)).squeeze()
            y = torch.tensor(y, dtype = torch.float).unsqueeze(0)

        
        return mu + (var**0.5) * (y - y.mean()) / y.std()

    return target_scaler



class Preprocessor():

    def __init__(
        self,
        N_datapoints: int,
        P_features: int,
        scale_features: callable = scale_features_01_power_transform,
        target_mean: float = 0.0,
        target_var: float = 1.0,
        power_transform_y: bool = True,
        seed: int = 0,
        additive_noise_std: float = 0.0
    ):
        """
        Args:
            N_datapoints (int): The number of datapoints to use.
            P_features (int): The number of features to use.
            scale_features (callable): A callable that scales the features.
            target_mean (float): The mean of the target.
            target_var (float): The variance of the target.
            power_transform_y (bool): Whether to apply a power transform to the target.
            seed (int): The seed to use.
            additive_noise_var (float): The variance of the additive noise.
        """
        self.N_datapoints = N_datapoints
        self.P_features = P_features
        self.scale_features = scale_features
        self.target_scaler = make_target_scaler(target_mean, target_var, power_transform = power_transform_y)

        self.seed = seed
        self.additive_noise_std = additive_noise_std

        # set the torch seed
        torch.manual_seed(seed)


    def _subsample_data(
        self,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """
        Subsample the data
        Args:
            x: torch.Tensor: the features
            y: torch.Tensor: the target
        Returns:
            x, y
        """

        assert len(x) >= self.N_datapoints, "The number of datapoints is larger than the number of datapoints in the dataset"
            
        indices = torch.randperm(x.shape[0])[:self.N_datapoints]

        x = x[indices]
        y = y[indices]

        return x, y
    
    def _identify_numerical_features(
            self,
            x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Identify the numerical features in the dataset
        Args:
            x: torch.Tensor: the features
        Returns:
            torch.Tensor: the features sorted by the number of different values
            n_diff_values: torch.Tensor: the number of different values for each feature
        """
        
        n_diff_values = torch.tensor([len(torch.unique(x[:, i])) for i in range(x.shape[1])])
        _, indices = torch.sort(n_diff_values, descending = True)

        x = x[:, indices]
        n_diff_values = n_diff_values[indices]

        return x, n_diff_values


    def preprocess(
            self,
            dataset: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """
        Scale the features and the target of the dataset
        Args:
            dataset: dict[str, torch.Tensor]: the dataset
        Returns:
            dict[str, torch.Tensor]: the preprocessed dataset
        """

        x = dataset["x"]
        y = dataset["y"]

        try:
            x = torch.tensor(x, dtype = torch.float)
            y = torch.tensor(y, dtype = torch.float)
        except Exception as e:
            raise ValueError("The features and target should be convertible to a torch tensor") from e

        assert len(x) == len(y), "The number of features and targets is different. got {} and {}".format(len(x), len(y))

        

        x, y = self._subsample_data(x, y)   

        #x = self.scale_features(x)
        #y = self.target_scaler(y)

        x, n_diff_values = self._identify_numerical_features(x)

        assert x.shape[1] >= self.P_features, "The number of features is smaller than the number of features to use"
        x = x[:, :self.P_features] # select the first P_features
        n_diff_values = n_diff_values[:self.P_features]

        if self.additive_noise_std > 0:
            y = y + torch.randn(y.shape) * self.additive_noise_std
            x = x + torch.randn(x.shape) * self.additive_noise_std

        x = self.scale_features(x)
        y = self.target_scaler(y)

        new_dataset = {
            "x": x,
            "y": y,
            "n_diff_values": n_diff_values
        }

        return new_dataset
    
class PreprocessorClassification(Preprocessor):

    def __init__(
        self,
        N_datapoints: int,
        P_features: int,
        scale_features: callable = scale_features_01_power_transform,
        seed: int = 0,
        additive_noise_std: float = 0.0
    ):
        """
        Args:
            N_datapoints (int): The number of datapoints to use.
            P_features (int): The number of features to use.
            scale_features (callable): A callable that scales the features.
            seed (int): The seed to use.
            additive_noise_var (float): The variance of the additive noise.
        """
        self.N_datapoints = N_datapoints
        self.P_features = P_features
        self.scale_features = scale_features

        self.seed = seed
        self.additive_noise_std = additive_noise_std

        # set the torch seed
        torch.manual_seed(seed)


    def preprocess(
            self,
            dataset: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """
        Scale the features and the target of the dataset
        Args:
            dataset: dict[str, torch.Tensor]: the dataset
        Returns:
            dict[str, torch.Tensor]: the preprocessed dataset
        """

        x = dataset["x"]
        y = dataset["y"]

        try:
            x = torch.tensor(x, dtype = torch.float)
            y = torch.tensor(y, dtype = torch.float)
        except Exception as e:
            raise ValueError("The features and target should be convertible to a torch tensor") from e

        assert len(x) == len(y), "The number of features and targets is different. got {} and {}".format(len(x), len(y))

        

        x, y = self._subsample_data(x, y)   

        #x = self.scale_features(x)
        #y = self.target_scaler(y)

        x, n_diff_values = self._identify_numerical_features(x)

        assert x.shape[1] >= self.P_features, "The number of features is smaller than the number of features to use"
        x = x[:, :self.P_features] # select the first P_features
        n_diff_values = n_diff_values[:self.P_features]
        if self.additive_noise_std > 0:
            x = x + torch.randn(x.shape) * self.additive_noise_std

        x = self.scale_features(x)

        new_dataset = {
            "x": x,
            "y": y,
            "n_diff_values": n_diff_values
        }

        return new_dataset
    

class PreprocessorGammaResponse():

    def __init__(
        self,
        N_datapoints: int,
        P_features: int,
        scale_features: callable = scale_features_01_power_transform,
        target_mean: float = 0.0,
        target_var: float = 1.0,
        target_lambda: float = 1.0,
        power_transform_y: bool = True,
        seed: int = 0,
        additive_noise_std: float = 0.0
    ):
        """
        Args:
            N_datapoints (int): The number of datapoints to use.
            P_features (int): The number of features to use.
            scale_features (callable): A callable that scales the features.
            target_mean (float): The mean of the target.
            target_var (float): The variance of the target.
            target_lambda (float): The lambda parameter of the boxcox transformation,
            power_transform_y (bool): Whether to apply a power transform to the target.
            seed (int): The seed to use.
            additive_noise_var (float): The variance of the additive noise.
        """
        self.N_datapoints = N_datapoints
        self.P_features = P_features
        self.scale_features = scale_features
        self.target_scaler = make_target_scaler(target_mean, target_var, power_transform = power_transform_y)

        self.seed = seed
        self.additive_noise_std = additive_noise_std
        self.target_lambda = target_lambda

        # set the torch seed
        torch.manual_seed(seed)


    def _subsample_data(
        self,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """
        Subsample the data
        Args:
            x: torch.Tensor: the features
            y: torch.Tensor: the target
        Returns:
            x, y
        """

        assert len(x) >= self.N_datapoints, "The number of datapoints is larger than the number of datapoints in the dataset"
            
        indices = torch.randperm(x.shape[0])[:self.N_datapoints]

        x = x[indices]
        y = y[indices]

        return x, y
    
    def _identify_numerical_features(
            self,
            x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Identify the numerical features in the dataset
        Args:
            x: torch.Tensor: the features
        Returns:
            torch.Tensor: the features sorted by the number of different values
            n_diff_values: torch.Tensor: the number of different values for each feature
        """
        
        n_diff_values = torch.tensor([len(torch.unique(x[:, i])) for i in range(x.shape[1])])
        _, indices = torch.sort(n_diff_values, descending = True)

        x = x[:, indices]
        n_diff_values = n_diff_values[indices]

        return x, n_diff_values


    def preprocess(
            self,
            dataset: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """
        Scale the features and the target of the dataset
        Args:
            dataset: dict[str, torch.Tensor]: the dataset
        Returns:
            dict[str, torch.Tensor]: the preprocessed dataset
        """

        x = dataset["x"]
        y = dataset["y"]

        try:
            x = torch.tensor(x, dtype = torch.float)
            y = torch.tensor(y, dtype = torch.float)
        except Exception as e:
            raise ValueError("The features and target should be convertible to a torch tensor") from e

        assert len(x) == len(y), "The number of features and targets is different. got {} and {}".format(len(x), len(y))

        

        x, y = self._subsample_data(x, y)   

        #x = self.scale_features(x)
        #y = self.target_scaler(y)

        x, n_diff_values = self._identify_numerical_features(x)

        assert x.shape[1] >= self.P_features, "The number of features is smaller than the number of features to use"
        x = x[:, :self.P_features] # select the first P_features
        n_diff_values = n_diff_values[:self.P_features]

        if self.additive_noise_std > 0:
            y = y + torch.randn(y.shape) * self.additive_noise_std
            x = x + torch.randn(x.shape) * self.additive_noise_std

        x = self.scale_features(x)
        

        y = torch.exp(y) # the target is the log of the response
        y = boxcox(y, self.target_lambda)

        y = torch.tensor(y, dtype = torch.float)

        y = self.target_scaler(y)

        new_dataset = {
            "x": x,
            "y": y,
            "n_diff_values": n_diff_values
        }

        return new_dataset