import torch 
import pyro 
import pyro.distributions as dist

import numpy as np
import numpyro
import numpyro.distributions as dist
import jax.numpy as jnp
from numpyro import plate
from jax import random

def make_lda_program(
        n_docs: int,
        n_words: int,
        n_topics: int,
        alpha_dir: float = 0.1,
        beta_dir: float = 0.1,
        doc_len_max: int = 100,
        doc_len_mean: int = 10,
        key = random.PRNGKey(0)
):
    """
    Make a probabilistic program for LDA in NumPyro.
    n_docs: int, number of documents
    n_words: int, number of words
    alpha_dir: float, Dirichlet prior for the document-topic distribution
    beta_dir: float, Dirichlet prior for the topic-word distribution
    doc_len_rate: float, rate parameter for the Poisson distribution for the document length
    """

    n_docs = int(n_docs)
    n_words = int(n_words)
    n_topics = int(n_topics)

    def lda_program(
            x: jnp.ndarray = None, 
            n_docs=n_docs,
            n_words=n_words,
            n_topics=n_topics,
            alpha_dir=alpha_dir,
            beta_dir=beta_dir,
            doc_len_max=doc_len_max,
            doc_len_mean=doc_len_mean,
    ):  
        
        rng_key, theta_key, phi_key, doc_len_key, x_key = random.split(key, 5)
        if x is not None:
            if type(x) == torch.Tensor:
                x = x.numpy()

            n_docs = x.shape[0]
            n_words = x.shape[1]

        theta_dist = dist.Dirichlet(alpha_dir * jnp.ones(n_topics))  # topic distribution per document
        phi_dist = dist.Dirichlet(beta_dir * jnp.ones(n_words))  # word distribution per topic
        doc_len_dist = dist.Binomial(doc_len_max, doc_len_mean / doc_len_max)  # document length

        # Sample theta and phi
        with plate("words", n_topics):
            phi = numpyro.sample("phi", phi_dist, rng_key=phi_key)

        with plate("docs", n_docs) as i:
            theta = numpyro.sample("theta", theta_dist, rng_key=theta_key)

            doc_len = numpyro.sample("doc_len", doc_len_dist, rng_key=doc_len_key)
            if x is not None:
                doc_len = x[i].sum()
            w_prod = jnp.matmul(theta, phi)

            x = numpyro.sample("x", dist.Multinomial(doc_len, w_prod), obs=x, rng_key=x_key)

        return {
            "theta": theta,
            "phi": phi,
            "doc_len": doc_len,
            "x": x
        }
    
    return lda_program


def make_lda_program_batched(
        n_docs: int,
        batch_size: int,
        n_words: int,
        n_topics: int,
        alpha_dir: float = 0.1,
        beta_dir: float = 0.1,
        doc_len_max: int = 100,
        doc_len_mean: int = 10,
):
    """
    Make a probabilistic program for LDA in NumPyro.
    n_docs: int, number of documents
    batch_size: int, batch size
    n_words: int, number of words
    alpha_dir: float, Dirichlet prior for the document-topic distribution
    beta_dir: float, Dirichlet prior for the topic-word distribution
    doc_len_rate: float, rate parameter for the Poisson distribution for the document length
    """

    n_docs = int(n_docs)
    n_words = int(n_words)
    n_topics = int(n_topics)
    batch_size = int(batch_size)

    def lda_program(
            x: jnp.ndarray = None, 
            n_docs=n_docs,
            batch_size = batch_size,
            n_words=n_words,
            n_topics=n_topics,
            alpha_dir=alpha_dir,
            beta_dir=beta_dir,
            doc_len_max=doc_len_max,
            doc_len_mean=doc_len_mean,
    ):  
        
        key = random.PRNGKey(torch.randint(0, 1000000, (1,)).item())
        rng_key, theta_key, phi_key, doc_len_key, x_key = random.split(key, 5)

        x_none = x is None
        if x_none is False:
            if type(x) == torch.Tensor:
                x = x.numpy()
            
            batch_size = x.shape[0]
            n_docs = x.shape[1]
            n_words = x.shape[2]

        theta_dist = dist.Dirichlet(alpha_dir * jnp.ones(n_topics))  # topic distribution per document
        phi_dist = dist.Dirichlet(beta_dir * jnp.ones(n_words))  # word distribution per topic
        doc_len_dist = dist.Binomial(doc_len_max, doc_len_mean / doc_len_max)  # document length

        with plate("batch", batch_size):
            # Sample theta and phi
            with plate("words", n_topics):
                phi = numpyro.sample("phi", phi_dist, rng_key=phi_key)

            with plate("docs", n_docs) as i:
                theta = numpyro.sample("theta", theta_dist, rng_key=theta_key)

                doc_len = numpyro.sample("doc_len", doc_len_dist, rng_key=doc_len_key)
                if x is not None:
                    doc_len = x[i].sum()
                
                theta = jnp.permute_dims(theta, (1,0,2))
                phi = jnp.permute_dims(phi, (1,0,2))
                w_prod = jnp.matmul(theta, phi)

                w_prod = jnp.permute_dims(w_prod, (1,0,2))

                del x
                x = numpyro.sample("x", dist.Multinomial(doc_len, w_prod), rng_key=x_key)

        x = jnp.permute_dims(x, (1,0,2))
        #theta = jnp.permute_dims(theta, (1,0,2))
        #phi = jnp.permute_dims(phi, (1,0,2))

        x = torch.tensor(np.asanyarray(x))
        theta = torch.tensor(np.asarray(theta))
        phi = torch.tensor(np.asarray(phi))

        beta = phi
        return {
            "theta": theta,
            "phi": phi,
            #"doc_len": doc_len,
            "x": x,
            "beta": beta
        }
    
    return lda_program