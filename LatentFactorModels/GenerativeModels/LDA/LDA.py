import torch 
import pyro 
import pyro.distributions as dist


def make_lda_pprogram(
        n_docs: int,
        n_words: int,
        alpha_dir: float = 0.1,
        beta_dir: float = 0.1,
        doc_len_rate: float = 10.0
):
    """
    Make a probabilistic program for LDA.
    n_docs: int, number of documents
    n_words: int, number of words
    alpha_dir: float, Dirichlet prior for the document-topic distribution
    beta_dir: float, Dirichlet prior for the topic-word distribution
    doc_len_rate: float, rate parameter for the Poisson distribution for the document length
    """

    n_docs = int(n_docs)
    n_words = int(n_words)

    def lda_program(
            x: torch.tensor = None, 
            n_docs = n_docs,
            n_words = n_words,
            alpha_dir = alpha_dir,
            beta_dir = beta_dir,
            doc_len_rate = doc_len_rate
    ):  
        
        if x is not None:
            n_docs = x.shape[0]
            n_words = x.shape[1]

        theta_dist = dist.Dirichlet(alpha_dir * torch.ones(n_words)) # document-topic distribution
        phi_dist = dist.Dirichlet(beta_dir * torch.ones(n_docs))  # topic-word distribution (aka beta)
        doc_len_dist = dist.Poisson(doc_len_rate)
        # sample theta and phi

        

        with pyro.plate("words", n_words):
            phi = pyro.sample("phi", phi_dist)

        with pyro.plate("docs", n_docs) as i:
            theta = pyro.sample("theta", theta_dist)

            doc_len = pyro.sample("doc_len", doc_len_dist)
            doc_len = x[i].sum()

            print(f"theta: {theta.shape}, phi: {phi.shape}, doc_len: {doc_len}")

            w_prod = torch.matmul(theta, phi)

            x = pyro.sample("x", dist.Multinomial(doc_len, w_prod), obs = x)

        return {
            "theta": theta,
            "phi": phi,
            "doc_len": doc_len,
            "x": x
        }
    
    return lda_program