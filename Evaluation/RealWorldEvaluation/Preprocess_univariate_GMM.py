import torch


class Preprocessor_GMM_univariate():

    def __init__(
        self,
        N_datapoints:int,
        x_mean: float = 0.0,
        x_var: float = 1.0,
    ):
        """
        Args:
            N: int: the number of data points
            x_mean: float: the mean of the features
            x_var: float: the variance of the features
        """
        self.N_datapoints = N_datapoints
        self.x_mean = x_mean
        self.x_var = x_var
   
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

        x = y

        # scale x to have x_mean and x_var as mean and variance
        x = (x - x.mean()) / (x.std() + 1e-8)
        x = x * self.x_var + self.x_mean

        x = x.reshape(-1, 1)

        new_dataset = {
            "x": x
        }

        return new_dataset