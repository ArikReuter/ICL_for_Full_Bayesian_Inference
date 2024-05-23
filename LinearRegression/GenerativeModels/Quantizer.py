import torch 
import math


class Quantizer():
    """
    A class that can be used to map continuous values to discrete values
    """
    def __init__(self, quantile_samples: torch.tensor, possible_bucket_numbers = [2**i for i in range(1, 10)]):
        """
        Args:
            quantile_samples: torch.tensor: a sample of the values to be quantized, used to compute quantiles for the quantization
            possible_bucket_numbers: List[int]: the possible number of buckets
        """
        assert len(quantile_samples.shape) == 1, "quantile_samples should be a 1D tensor"

        self.quantile_samples = quantile_samples
        self.possible_bucket_numbers = possible_bucket_numbers

        self.max_number_buckets = math.lcm(*possible_bucket_numbers)

        self.quantiles = torch.quantile(quantile_samples, torch.linspace(0, 1, self.max_number_buckets + 1))

    
    def quantize(self, x: torch.tensor, n_buckets: int) -> torch.tensor:
        """
        Perform quantization on a tensor by keeping the original values and mapping them to the closest quantile value
        Args:
            x: torch.tensor: the tensor to quantize
            n_buckets: int: the number of buckets to use
        """
        assert n_buckets in self.possible_bucket_numbers, f"n_buckets should be in {self.possible_bucket_numbers}"

        quantiles = self.quantiles[::(self.max_number_buckets // n_buckets)]
        quantile_means = (quantiles[1:] + quantiles[:-1]) / 2
        quantized = torch.zeros_like(x)

        for i in range(n_buckets):
            quantized += ((x >= quantiles[i]) & (x < quantiles[i + 1])) * quantile_means[i]

        return quantized

