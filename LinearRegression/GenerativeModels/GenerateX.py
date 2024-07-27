import torch 

def simulate_X_uniform(n:int, p:int, batch_size:int = 0) -> torch.tensor:
    """
    Simulate covariates from a uniform distribution on the unit interval
    Args:
        n: int: the number of observations
        p: int: the number of covariates
        batch_size: int: the batch size

    Returns:
        torch.tensor: the covariates of shape (n, p) or (batch_size, n, p)
    """

    if batch_size == 0:
        return torch.rand(n, p)
    else:
        return torch.rand(batch_size, n, p)
    


def simulate_X_uniform_discretized(
        n:int,
        p:int,
        batch_size:int = 1,
        
):
    """
    Simulate covariates from a uniform distribution on the unit interval tha are discretized
    Args:
        n: int: the number of observations
        p: int: the number of covariates
        batch_size: int: the batch size
        n_diff_values_dist: torch.distributions.Categorical: the distribution of the number of different values
    """

    probs = torch.rand(batch_size, p, n)

    if n == 100:
        n1 = int(n*0.2)
        n2 = n - n1 - 1
        prob_vec = [0]*n1 + [1]*n2 + [10]
        prob_vec = torch.tensor(prob_vec)
    

        print(prob_vec)

    else:
        n1 = int(n*0.5)
        n2 = n - n1 - 1
        prob_vec = [0]*n1 + [1]*n2 + [10]
        prob_vec = torch.tensor(prob_vec)

    n_diff_values_dist = torch.distributions.Categorical(prob_vec)

    #randints = torch.distributions.Categorical(probs).sample((n,)).permute(1,0,2)
    

    randints = torch.randint(0, 99, (batch_size, n, p)) + 1
    # randomly permute the integers along the n dimension



    n_diff_values = n_diff_values_dist.sample((batch_size, p)) + 1


    new_randints = randints % n_diff_values.unsqueeze(1) #shape (batch_size, n, p)

    R = torch.rand(batch_size, n, p)

    # Create the indices for the batch dimension and last dimension
    batch_indices = torch.arange(batch_size).unsqueeze(1).unsqueeze(2).expand(batch_size, n, p)
    last_indices = torch.arange(p).unsqueeze(0).unsqueeze(0).expand(batch_size, n, p)

    # Use advanced indexing to map the values
    P = R[batch_indices, new_randints, last_indices]

    return P, new_randints



def simulate_X_uniform_discretized2(
        n:int,
        p:int,
        batch_size:int = 1,
        
):
    """
    Simulate covariates from a uniform distribution on the unit interval tha are discretized
    Args:
        n: int: the number of observations
        p: int: the number of covariates
        batch_size: int: the batch size
        n_diff_values_dist: torch.distributions.Categorical: the distribution of the number of different values
    """

    if n == 100:
        n1 = int(n*0.2)
        n2 = n - n1 -1
        prob_vec = [0]*n1 + [1]*n2 + [10]
        prob_vec = torch.tensor(prob_vec)
    

        print(len(prob_vec))

    else:
        n1 = int(n*0.5)
        n2 = n - n1 -1
        prob_vec = [0]*n1 + [1]*n2 + [10]
        prob_vec = torch.tensor(prob_vec)


    n_diff_values_dist = torch.distributions.Categorical(prob_vec)

    res = torch.rand(batch_size, n, p)

    for i in range(batch_size):
        for j in range(p):
            n_diff_values = n_diff_values_dist.sample() +1

            randperm = torch.randperm(n) % n_diff_values

            res[i, :, j] = res[i, randperm, j]

    return res