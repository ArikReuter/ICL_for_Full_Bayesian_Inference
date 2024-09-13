import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro import sample, plate
from jax import random

def make_fa_program_normal_weight_prior(
        n: int = 100,
        p: int = 5,
        z_dim: int = 3,
        w_var: float = 10.0,
        mu_var: float = 0.1,
        a1_psi_var: float = 5.0,
        b1_psi_var: float = 2.0,
        key = random.PRNGKey(0)
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
    rng_key = key

    def fa_program(x: jnp.ndarray = None, rng_key = rng_key, p=p, n=n) -> dict:
        """
        A univariate GMM program

        Args:
            rng_key: PRNG key for sampling
            x: jnp.ndarray: the input

        Returns:
            dict: the output
        """
        if x is not None:
            n = x.shape[0]
            p = x.shape[1]

        if x is not None:
            x = x.squeeze()

        n_weights = p * z_dim - (z_dim * (z_dim - 1)) // 2

        rng_key, z_key, mu_key, w_key, psi_key = random.split(rng_key, 5)

        z_dist = dist.MultivariateNormal(jnp.zeros(z_dim), jnp.eye(z_dim))
        mu_dist = dist.MultivariateNormal(jnp.zeros(p), mu_var * jnp.eye(p))
        w_dist = dist.Normal(0, w_var)

        z = sample("z", z_dist, rng_key=z_key)  # shape: (p,)
        mu = sample("mu", mu_dist, rng_key=mu_key)

        # Generate W where only the lower triangular part is non-zero
        with plate("weights", n_weights):
            w = sample("w", w_dist, rng_key=w_key)

        W = jnp.zeros((p, z_dim))

        tri_indices = jnp.tril_indices(n = p, m = z_dim)
        W = W.at[tri_indices[0], tri_indices[1]].set(w)

        # Also make the diagonal elements positive
        diagonal_elements = jnp.diag(W)
        diagonal_abs = jnp.abs(diagonal_elements)
        diag_size = min(p, z_dim)
        new_matrix = W.at[jnp.diag_indices(diag_size)].set(diagonal_abs[:diag_size])

        W = new_matrix

        psi_diag_dist = dist.InverseGamma(a1_psi_var, b1_psi_var)
        with plate("n_dims", p):
            psi_diag = sample("psi", psi_diag_dist, rng_key=psi_key)

        psi = jnp.diag(psi_diag)

        mean_x = mu + W @ z
        x_dist = dist.MultivariateNormal(mean_x, psi)

        with plate("data", n):
            x = sample("x", x_dist, obs=x, rng_key=rng_key)

        return {
            "x": x,
            "z": z,
            "mu": mu,
            "w": w,
            "psi": psi_diag
        }

    return fa_program
