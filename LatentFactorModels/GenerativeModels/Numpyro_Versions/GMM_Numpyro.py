

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax.random import PRNGKey
import jax

def make_gmm_program_diagonal(
    n: int = 100,
    p: int = 5,
    k: int = 3,
    a1: float = 5.0,
    b1: float = 2.0,
    dirichlet_beta: float = 1.0,
    lambda1: float = 3.0,
    key = PRNGKey(0)
):
    """
    Make a probabilistic program for a univariate GMM.

    Args:
        n (int): Number of data points.
        p (int): Number of dimensions.
        k (int): Number of components.
        a1 (float): Shape of inverse gamma prior on sigma_squared.
        b1 (float): Scale of inverse gamma prior on sigma_squared.
        dirichlet_beta (float): Concentration parameter of the Dirichlet prior on the mixture weights.
        lambda1 (float): Scale the variance for the prior on the mus.
    """
    a1 = jnp.array(a1)
    b1 = jnp.array(b1)
    dirichlet_beta = jnp.array(dirichlet_beta)
    lambda1 = jnp.array(lambda1)

    p = int(p)
    n = int(n)
    k = int(k)


    def gmm_program(x: jnp.array = None, prng_key = key, p=p, k=k, n=n) -> dict:
        """
        A univariate GMM program.

        Args:
            x (jnp.array): The input data.

        Returns:
            dict: A dictionary containing the sampled variables.
        """

        key_phi, key_mu, key_sigma_squared, key_z, key_x = jax.random.split(prng_key, 5)
        if x is not None:
            x = jnp.squeeze(x)
            #print(f"x: {x.shape}")

        # Mixture weights
        phi = numpyro.sample("phi", dist.Dirichlet(jnp.ones(k) * dirichlet_beta), rng_key=key_phi)

        # Component parameters


        with numpyro.plate("features", p):
            with numpyro.plate("components", k):
                sigma_squared = numpyro.sample(
                    "sigma_squared",
                    dist.InverseGamma(a1, b1),
                    rng_key = key_sigma_squared
                )
                mu = numpyro.sample(
                    "mu",
                    dist.Normal(0, jnp.sqrt(lambda1 * sigma_squared)),
                    rng_key = key_mu
                )

        # Data likelihood
        with numpyro.plate("data", n):
            z = numpyro.sample("z", dist.Categorical(probs=phi), rng_key=key_z)
            #print(f"z: {z.shape}")
            #print(f"mu: {mu.shape}")
            #print(f"sigma_squared: {sigma_squared.shape}")
    
            mu_z = mu[z]
            sigma_squared_z = sigma_squared[z]

            # permute last two dimensions of mu_z
            #mu_z = jnp.moveaxis(mu_z, 0, 1)
            #sigma_squared_z = jnp.moveaxis(sigma_squared_z, 0, 1)

            #print(f"mu_z: {mu_z.shape}")
            #print(f"sigma_squared_z: {sigma_squared_z.shape}")

            # create batched diagonal covariance matrix

            x = numpyro.sample(
                    "x",
                    dist.Normal(mu_z, jnp.sqrt(sigma_squared_z)).to_event(1),
                    obs=x,
                    rng_key=key_x
            )


        return {
            "phi": phi,
            "mu": mu,

            "sigma_squared": sigma_squared,
            "z": z,
            "x": x,
        }

    return gmm_program
