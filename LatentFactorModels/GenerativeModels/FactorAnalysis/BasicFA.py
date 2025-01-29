import torch 
import pyro 
import pyro.distributions as dist

def make_fa_program_normal_weight_prior(
        n: int = 100,
        p: int = 5,
        z_dim: int = 3,
        w_var: float = 10.0,
        mu_var: float = 0.1,
        a1_psi_var: float = 5.0,
        b1_psi_var: float = 2.0,
):
    """
    Make a probabilistic program for a univariate GMM 
    Args:   
        n: int: number of data points
        p: int: number of dimensions of data
        z_dim: int: number of factors
        w_var: float: variance of the weights
        mu_var: float: variance of the mean
        a1_psi_var: float: parameter for the InverseGamma distribution
        b1_psi_var: float: parameter for the InverseGamma distribution

    Returns:
        function: a probabilistic program
    """
    n = int(n)
    p = int(p)
    z_dim = int(z_dim)
    
       #n_weights = p*z_dim - (z_dim * (z_dim - 1)) //2
    
    def fa_program(x: torch.tensor = None, p = p, n = n) -> dict:
        """
        A univariate GMM program

        Args:
            x: torch.tensor: the input

        Returns:
            torch.tensor: the output
        """
        if x is not None:
            n = x.shape[0]
            p = x.shape[1]  
        
        if x is not None:
            x = x.squeeze()

        n_weights = p*z_dim - (z_dim * (z_dim - 1)) //2

        z_dist = dist.MultivariateNormal(torch.zeros(z_dim), torch.eye(z_dim))
        mu_dist = dist.MultivariateNormal(torch.zeros(p), mu_var * torch.eye(p))
        
        w_dist = dist.Normal(0, w_var)

        z = pyro.sample("z", z_dist)  # shape: (p,)
        mu = pyro.sample("mu", mu_dist)

        # generate W where only the lower triangular part is non-zero
        with pyro.plate("weights", n_weights):
            w = pyro.sample("w", w_dist)

        W = torch.zeros(p, z_dim)
        tri_indices = torch.tril_indices(row=p, col=z_dim)
        W[tri_indices[0], tri_indices[1]] = w

        # also make the diagonal elements positive
        diagonal_elements = torch.diag(W)
        diagonal_abs = torch.abs(diagonal_elements)
        diag_size = min(p, z_dim)
        new_matrix = W.clone()
        new_matrix[range(diag_size), range(diag_size)] = diagonal_abs[:diag_size]

        W = new_matrix

        psi_diag_dist = dist.InverseGamma(a1_psi_var, b1_psi_var)
        with pyro.plate("n_dims",p):
            psi_diag = pyro.sample("psi", psi_diag_dist)
        
        psi_diag += 1e-6

        psi = torch.diag(psi_diag) 

        mean_x = mu + W @ z
        x_dist = dist.MultivariateNormal(mean_x, psi)

        with pyro.plate("data", n):
            x = pyro.sample("x", x_dist, obs=x)

        return {
            "x": x,
            "z": z,
            "mu": mu,
            "w": w,
            "psi": psi_diag
        }
    
    return fa_program	
    

def make_fa_program_normal_weight_prior_batched(
        n: int = 100,
        p: int = 5,
        batch_size: int = 32,
        z_dim: int = 3,
        w_var: float = 0.1,
        mu_var: float = 3.0,
        a1_psi_var: float = 5.0,
        b1_psi_var: float = 2.0,
):
    """
    Make a probabilistic program for a univariate GMM 
    Args:   
        n: int: number of data points
        p: int: number of dimensions of data
        z_dim: int: number of factors
        w_var: float: variance of the weights
        mu_var: float: variance of the mean
        a1_psi_var: float: parameter for the InverseGamma distribution
        b1_psi_var: float: parameter for the InverseGamma distribution

    Returns:
        function: a probabilistic program
    """
    n = int(n)
    p = int(p)
    z_dim = int(z_dim)
    batch_size = int(batch_size)
    
    
    
    def fa_program(x: torch.tensor = None, batch_size = batch_size, p = p, n = n) -> dict:
        """
        A univariate GMM program

        Args:
            x: torch.tensor: the input

        Returns:
            torch.tensor: the output
        """


        if x is not None:
            batch_size = x.shape[0]
            n = x.shape[1]
            p = x.shape[2]  

        n_weights = p*z_dim - (z_dim * (z_dim - 1)) //2

        if x is not None:
            x = x.squeeze()

        z_dist = dist.MultivariateNormal(torch.zeros(z_dim), torch.eye(z_dim))
        mu_dist = dist.MultivariateNormal(torch.zeros(p), mu_var * torch.eye(p))
        
        w_dist = dist.Normal(0, w_var)
        
        with pyro.plate("batch", batch_size):

            z = pyro.sample("z", z_dist)  # shape: (p,)
            mu = pyro.sample("mu", mu_dist)

            # generate W where only the lower triangular part is non-zero
            with pyro.plate("weights", n_weights):
                w = pyro.sample("w", w_dist)
            
            w = w.T

        W = torch.zeros(batch_size, p, z_dim)
        tri_indices = torch.tril_indices(row=p, col=z_dim)
        W[:, tri_indices[0], tri_indices[1]] = w

        # also make the diagonal elements positive
        diag_size = min(p, z_dim)
        diagonal_elements = W[:, range(diag_size), range(diag_size)]

        diagonal_abs = torch.abs(diagonal_elements)
        
        new_matrix = W.clone()
        new_matrix[:, range(diag_size), range(diag_size)] = diagonal_abs

        W = new_matrix

        psi_diag_dist = dist.InverseGamma(a1_psi_var, b1_psi_var)

        with pyro.plate("batch", batch_size):
            with pyro.plate("n_dims", p):
                psi_diag = pyro.sample("psi", psi_diag_dist)

        psi = torch.diag_embed(psi_diag.T)

        # print all shapes 


        mean_x = mu + torch.bmm(W, z.unsqueeze(-1)).squeeze(-1)
        
        mean_x = mean_x.unsqueeze(1).repeat(1, n, 1)
        psi = psi.unsqueeze(1).repeat(1, n, 1, 1)

        x_dist = dist.MultivariateNormal(mean_x, psi)

        x = pyro.sample("x", x_dist)

        beta = z


        return {
            "x": x,
            "z": z,
            "mu": mu,
            "w": w,
            "psi": psi_diag.T,
            "beta": beta
        }



    return fa_program


def make_fa_program_laplace_weight_prior(
        n: int = 100,
        p: int = 5,
        z_dim: int = 3,
        w_var: float = 10.0,
        mu_var: float = 0.1,
        a1_psi_var: float = 5.0,
        b1_psi_var: float = 2.0,
):
    """
    Make a probabilistic program for a univariate GMM 
    Args:   
        n: int: number of data points
        p: int: number of dimensions of data
        z_dim: int: number of factors
        w_var: float: variance of the weights
        mu_var: float: variance of the mean
        a1_psi_var: float: parameter for the InverseGamma distribution
        b1_psi_var: float: parameter for the InverseGamma distribution

    Returns:
        function: a probabilistic program
    """
    n = int(n)
    p = int(p)
    z_dim = int(z_dim)
    
       #n_weights = p*z_dim - (z_dim * (z_dim - 1)) //2
    
    def fa_program(x: torch.tensor = None, p = p, n = n) -> dict:
        """
        A univariate GMM program

        Args:
            x: torch.tensor: the input

        Returns:
            torch.tensor: the output
        """
        if x is not None:
            n = x.shape[0]
            p = x.shape[1]  
        
        if x is not None:
            x = x.squeeze()

        n_weights = p*z_dim - (z_dim * (z_dim - 1)) //2

        z_dist = dist.MultivariateNormal(torch.zeros(z_dim), torch.eye(z_dim))
        mu_dist = dist.MultivariateNormal(torch.zeros(p), mu_var * torch.eye(p))
        
        w_dist = dist.Laplace(0, w_var)

        z = pyro.sample("z", z_dist)  # shape: (p,)
        mu = pyro.sample("mu", mu_dist)

        # generate W where only the lower triangular part is non-zero
        with pyro.plate("weights", n_weights):
            w = pyro.sample("w", w_dist)

        W = torch.zeros(p, z_dim)
        tri_indices = torch.tril_indices(row=p, col=z_dim)
        W[tri_indices[0], tri_indices[1]] = w

        # also make the diagonal elements positive
        diagonal_elements = torch.diag(W)
        diagonal_abs = torch.abs(diagonal_elements)
        diag_size = min(p, z_dim)
        new_matrix = W.clone()
        new_matrix[range(diag_size), range(diag_size)] = diagonal_abs[:diag_size]

        W = new_matrix

        psi_diag_dist = dist.InverseGamma(a1_psi_var, b1_psi_var)
        with pyro.plate("n_dims",p):
            psi_diag = pyro.sample("psi", psi_diag_dist)

        psi = torch.diag(psi_diag)

        mean_x = mu + W @ z
        x_dist = dist.MultivariateNormal(mean_x, psi)

        with pyro.plate("data", n):
            x = pyro.sample("x", x_dist, obs=x)

        return {
            "x": x,
            "z": z,
            "mu": mu,
            "w": w,
            "psi": psi_diag
        }



    return fa_program


def make_fa_program_laplace_weight_prior_batched(
        n: int = 100,
        p: int = 5,
        batch_size: int = 32,
        z_dim: int = 3,
        w_var: float = 0.1,
        mu_var: float = 3.0,
        a1_psi_var: float = 5.0,
        b1_psi_var: float = 2.0,
):
    """
    Make a probabilistic program for a univariate GMM 
    Args:   
        n: int: number of data points
        p: int: number of dimensions of data
        z_dim: int: number of factors
        w_var: float: variance of the weights
        mu_var: float: variance of the mean
        a1_psi_var: float: parameter for the InverseGamma distribution
        b1_psi_var: float: parameter for the InverseGamma distribution

    Returns:
        function: a probabilistic program
    """
    n = int(n)
    p = int(p)
    z_dim = int(z_dim)
    batch_size = int(batch_size)
    
    
    
    def fa_program(x: torch.tensor = None, batch_size = batch_size, p = p, n = n) -> dict:
        """
        A univariate GMM program

        Args:
            x: torch.tensor: the input

        Returns:
            torch.tensor: the output
        """


        if x is not None:
            batch_size = x.shape[0]
            n = x.shape[1]
            p = x.shape[2]  

        n_weights = p*z_dim - (z_dim * (z_dim - 1)) //2

        if x is not None:
            x = x.squeeze()

        z_dist = dist.MultivariateNormal(torch.zeros(z_dim), torch.eye(z_dim))
        mu_dist = dist.MultivariateNormal(torch.zeros(p), mu_var * torch.eye(p))
        
        w_dist = dist.Laplace(0, w_var)
        
        with pyro.plate("batch", batch_size):

            z = pyro.sample("z", z_dist)  # shape: (p,)
            mu = pyro.sample("mu", mu_dist)

            # generate W where only the lower triangular part is non-zero
            with pyro.plate("weights", n_weights):
                w = pyro.sample("w", w_dist)
            
            w = w.T

        W = torch.zeros(batch_size, p, z_dim)
        tri_indices = torch.tril_indices(row=p, col=z_dim)
        W[:, tri_indices[0], tri_indices[1]] = w

        # also make the diagonal elements positive
        diag_size = min(p, z_dim)
        diagonal_elements = W[:, range(diag_size), range(diag_size)]

        diagonal_abs = torch.abs(diagonal_elements)
        
        new_matrix = W.clone()
        new_matrix[:, range(diag_size), range(diag_size)] = diagonal_abs

        W = new_matrix

        psi_diag_dist = dist.InverseGamma(a1_psi_var, b1_psi_var)

        with pyro.plate("batch", batch_size):
            with pyro.plate("n_dims", p):
                psi_diag = pyro.sample("psi", psi_diag_dist)

        psi = torch.diag_embed(psi_diag.T)

        # print all shapes 


        mean_x = mu + torch.bmm(W, z.unsqueeze(-1)).squeeze(-1)
        
        mean_x = mean_x.unsqueeze(1).repeat(1, n, 1)
        psi = psi.unsqueeze(1).repeat(1, n, 1, 1)

        x_dist = dist.MultivariateNormal(mean_x, psi)

        x = pyro.sample("x", x_dist)

        beta = z


        return {
            "x": x,
            "z": z,
            "mu": mu,
            "w": w,
            "psi": psi_diag.T,
            "beta": beta
        }



    return fa_program



def make_fa_program_normal_weight_prior_laplace_z_prior(
        n: int = 100,
        p: int = 5,
        z_dim: int = 3,
        w_var: float = 10.0,
        mu_var: float = 0.1,
        a1_psi_var: float = 5.0,
        b1_psi_var: float = 2.0,
):
    """
    Make a probabilistic program for a univariate GMM 
    Args:   
        n: int: number of data points
        p: int: number of dimensions of data
        z_dim: int: number of factors
        w_var: float: variance of the weights
        mu_var: float: variance of the mean
        a1_psi_var: float: parameter for the InverseGamma distribution
        b1_psi_var: float: parameter for the InverseGamma distribution

    Returns:
        function: a probabilistic program
    """
    n = int(n)
    p = int(p)
    z_dim = int(z_dim)
    
       #n_weights = p*z_dim - (z_dim * (z_dim - 1)) //2
    
    def fa_program(x: torch.tensor = None, p = p, n = n) -> dict:
        """
        A univariate GMM program

        Args:
            x: torch.tensor: the input

        Returns:
            torch.tensor: the output
        """
        if x is not None:
            n = x.shape[0]
            p = x.shape[1]  
        
        if x is not None:
            x = x.squeeze()

        n_weights = p*z_dim - (z_dim * (z_dim - 1)) //2

        z_dist = dist.Independent(dist.Laplace(torch.zeros(z_dim), torch.ones(z_dim)), 1)
        mu_dist = dist.MultivariateNormal(torch.zeros(p), mu_var * torch.eye(p))
        
        w_dist = dist.Normal(0, w_var)

        z = pyro.sample("z", z_dist)  # shape: (p,)
        mu = pyro.sample("mu", mu_dist)

        # generate W where only the lower triangular part is non-zero
        with pyro.plate("weights", n_weights):
            w = pyro.sample("w", w_dist)

        W = torch.zeros(p, z_dim)
        tri_indices = torch.tril_indices(row=p, col=z_dim)
        W[tri_indices[0], tri_indices[1]] = w

        # also make the diagonal elements positive
        diagonal_elements = torch.diag(W)
        diagonal_abs = torch.abs(diagonal_elements)
        diag_size = min(p, z_dim)
        new_matrix = W.clone()
        new_matrix[range(diag_size), range(diag_size)] = diagonal_abs[:diag_size]

        W = new_matrix

        psi_diag_dist = dist.InverseGamma(a1_psi_var, b1_psi_var)
        with pyro.plate("n_dims",p):
            psi_diag = pyro.sample("psi", psi_diag_dist)

        psi = torch.diag(psi_diag)

        mean_x = mu + W @ z
        x_dist = dist.MultivariateNormal(mean_x, psi)

        with pyro.plate("data", n):
            x = pyro.sample("x", x_dist, obs=x)

        return {
            "x": x,
            "z": z,
            "mu": mu,
            "w": w,
            "psi": psi_diag
        }
    
    return fa_program	
    

def make_fa_program_normal_weight_prior_laplace_z_prior_batched(
        n: int = 100,
        p: int = 5,
        batch_size: int = 32,
        z_dim: int = 3,
        w_var: float = 0.1,
        mu_var: float = 3.0,
        a1_psi_var: float = 5.0,
        b1_psi_var: float = 2.0,
):
    """
    Make a probabilistic program for a univariate GMM 
    Args:   
        n: int: number of data points
        p: int: number of dimensions of data
        z_dim: int: number of factors
        w_var: float: variance of the weights
        mu_var: float: variance of the mean
        a1_psi_var: float: parameter for the InverseGamma distribution
        b1_psi_var: float: parameter for the InverseGamma distribution

    Returns:
        function: a probabilistic program
    """
    n = int(n)
    p = int(p)
    z_dim = int(z_dim)
    batch_size = int(batch_size)
    
    
    
    def fa_program(x: torch.tensor = None, batch_size = batch_size, p = p, n = n) -> dict:
        """
        A univariate GMM program

        Args:
            x: torch.tensor: the input

        Returns:
            torch.tensor: the output
        """


        if x is not None:
            batch_size = x.shape[0]
            n = x.shape[1]
            p = x.shape[2]  

        n_weights = p*z_dim - (z_dim * (z_dim - 1)) //2

        if x is not None:
            x = x.squeeze()

        z_dist = dist.Independent(dist.Laplace(torch.zeros(z_dim), torch.ones(z_dim)), 1)
        mu_dist = dist.MultivariateNormal(torch.zeros(p), mu_var * torch.eye(p))
        
        w_dist = dist.Normal(0, w_var)
        
        with pyro.plate("batch", batch_size):

            z = pyro.sample("z", z_dist)  # shape: (p,)
            mu = pyro.sample("mu", mu_dist)

            # generate W where only the lower triangular part is non-zero
            with pyro.plate("weights", n_weights):
                w = pyro.sample("w", w_dist)
            
            w = w.T

        W = torch.zeros(batch_size, p, z_dim)
        tri_indices = torch.tril_indices(row=p, col=z_dim)
        W[:, tri_indices[0], tri_indices[1]] = w

        # also make the diagonal elements positive
        diag_size = min(p, z_dim)
        diagonal_elements = W[:, range(diag_size), range(diag_size)]

        diagonal_abs = torch.abs(diagonal_elements)
        
        new_matrix = W.clone()
        new_matrix[:, range(diag_size), range(diag_size)] = diagonal_abs

        W = new_matrix

        psi_diag_dist = dist.InverseGamma(a1_psi_var, b1_psi_var)

        with pyro.plate("batch", batch_size):
            with pyro.plate("n_dims", p):
                psi_diag = pyro.sample("psi", psi_diag_dist)

        psi = torch.diag_embed(psi_diag.T)

        # print all shapes 


        mean_x = mu + torch.bmm(W, z.unsqueeze(-1)).squeeze(-1)
        
        mean_x = mean_x.unsqueeze(1).repeat(1, n, 1)
        psi = psi.unsqueeze(1).repeat(1, n, 1, 1)

        x_dist = dist.MultivariateNormal(mean_x, psi)

        x = pyro.sample("x", x_dist)

        beta = z


        return {
            "x": x,
            "z": z,
            "mu": mu,
            "w": w,
            "psi": psi_diag.T,
            "beta": beta
        }



    return fa_program