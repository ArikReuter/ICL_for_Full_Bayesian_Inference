import torch 
from sklearn.preprocessing import PowerTransformer

class TargetScaler:
    """
    A class that transforms programs such that the target is scaled.
    """
    def __init__(self, transform: str = "standardize", variable_to_scale: str = "y"):
        """
        Args:
            transform (str): The transformation to apply to the target.
            variable_to_scale (str): The variable to scale.
        """
        assert transform in ["standardize", "power"], "The transformation is not recognized"
        self.transform = transform
        self.variable_to_scale = variable_to_scale

    def _scale_y(self, y: torch.Tensor) -> torch.Tensor:
        """
        Scale the target.
        
        Args:
            y (torch.Tensor): The target of shape (N,).
        
        Returns:
            torch.Tensor: The scaled target.
        """
        assert len(y.shape) == 1, "The target should be a vector"
        
        if self.transform == "standardize":
            return (y - y.mean()) / y.std()
        elif self.transform == "power":
            pt = PowerTransformer()
            return torch.tensor(pt.fit_transform(y.reshape(-1, 1)).flatten(), dtype=y.dtype)
        else:
            raise ValueError("The transformation is not recognized")
        
    def _scale_y_batched(self, y: torch.Tensor) -> torch.Tensor:
        """
        Scale the target in batches.
        
        Args:
            y (torch.Tensor): The target of shape (Batch_size, N).
        
        Returns:
            torch.Tensor: The scaled target.
        """
        assert len(y.shape) == 2, "The target should be a matrix"
        
        if self.transform == "standardize":
            return (y - y.mean(dim=1, keepdim=True)) / y.std(dim=1, keepdim=True)
        elif self.transform == "power":
            pt = PowerTransformer()
            y_new_list = [torch.tensor(pt.fit_transform(y[i].reshape(-1, 1)).flatten(), dtype=y.dtype) for i in range(y.shape[0])]
            return torch.stack(y_new_list)
        else:
            raise ValueError("The transformation is not recognized")
        
    def transform_y(self, pprogram_maker: callable) -> callable:
        """
        Transform the target of a pprogram that is made by a pprogram_maker
        Args:
            pprogram_maker: callable: a pprogram maker
        Returns:
            callable: the transformed pprogram maker
        """

        def transformed_pprogram_maker(*args_outer, **kwargs_outer):
            pprogram = pprogram_maker(*args_outer, **kwargs_outer)
            def transformed_pprogram(*args_inner, **kwargs_inner):
                pprogram_return_dict = pprogram(*args_inner, **kwargs_inner)
                y = pprogram_return_dict[self.variable_to_scale]
                pprogram_return_dict[self.variable_to_scale] = self._scale_y(y)

                return pprogram_return_dict
            
            return transformed_pprogram
        
        return transformed_pprogram_maker

    def transform_y_batched(self, pprogram_maker_batched: callable) -> callable:
        """
        Transform the target of a pprogram that is made by a pprogram_maker
        Args:
            pprogram_maker_batched: callable: a pprogram maker
        Returns:
            callable: the transformed pprogram maker
        """

        def transformed_pprogram_maker_batched(*args_outer, **kwargs_outer):
            pprogram = pprogram_maker_batched(*args_outer, **kwargs_outer)
            def transformed_pprogram(*args_inner, **kwargs_inner):
                pprogram_return_dict = pprogram(*args_inner, **kwargs_inner)
                y = pprogram_return_dict[self.variable_to_scale]
                pprogram_return_dict[self.variable_to_scale] = self._scale_y_batched(y)

                return pprogram_return_dict
            
            return transformed_pprogram
        
        return transformed_pprogram_maker_batched