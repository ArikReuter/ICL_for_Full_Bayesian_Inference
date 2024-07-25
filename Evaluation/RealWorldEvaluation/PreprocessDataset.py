import torch 

def scale_features_01(X):
    return 0.95*(X - X.min()) / (X.max() - X.min()) + 0.025  # scale between 0.025 and 0.975

def make_target_scaler(mu: float = 0.0, var: float = 1.0) -> callable:
    """
    Make a target scaler that scales the target to have mean mu and variance var
    Args:
        mu: float: the mean of the target
        var: float: the variance of the target
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
        return mu + (var**0.5) * (y - y.mean()) / y.std()

    return target_scaler



class Preprocessor():

    def __init__(
        self,
        N_datapoints: int,
        P_features: int,
        scale_features: callable = scale_features_01,
        target_mean: float = 0.0,
        target_var: float = 1.0,
        seed: int = 0
    ):
        """
        Args:
            N_datapoints (int): The number of datapoints to use.
            P_features (int): The number of features to use.
            scale_features (callable): A callable that scales the features.
            target_mean (float): The mean of the target.
            target_var (float): The variance of the target.
        """
        self.N_datapoints = N_datapoints
        self.P_features = P_features
        self.scale_features = scale_features
        self.target_scaler = make_target_scaler(target_mean, target_var)
        self.seed = seed

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
    ) -> torch.Tensor:
        """
        Identify the numerical features in the dataset
        Args:
            x: torch.Tensor: the features
        Returns:
            torch.Tensor: the features sorted by the number of different values
        """
        
        n_diff_values = torch.tensor([len(torch.unique(x[:, i])) for i in range(x.shape[1])])
        _, indices = torch.sort(n_diff_values, descending = True)

        x = x[:, indices]

        return x


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

        x = self.scale_features(x)
        y = self.target_scaler(y)

        x, y = self._subsample_data(x, y)   

        x = self._identify_numerical_features(x)

        assert x.shape[1] >= self.P_features, "The number of features is smaller than the number of features to use"
        x = x[:, :self.P_features] # select the first P_features

        new_dataset = {
            "x": x,
            "y": y
        }

        return new_dataset