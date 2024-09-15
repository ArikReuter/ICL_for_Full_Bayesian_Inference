from PFNExperiments.LatentFactorModels.ComparisonModels.Hamiltionian_MC import Hamiltionian_MC
from pyro.infer.autoguide import AutoDiagonalNormal, AutoMultivariateNormal, AutoLaplaceApproximation, AutoIAFNormal, AutoStructured
from PFNExperiments.LatentFactorModels.ComparisonModels.Variational_InferenceAutoguide import Variational_InferenceAutoguide
from PFNExperiments.LatentFactorModels.ComparisonModels.Hamiltionian_MC_Numpyro import Hamiltionian_MC as Hamiltionian_MC_Numpyro


def make_default_list_comparison(
        pprogram,
        n_samples: int = 1000,
        discrete_z: bool = True
        ):
    """
    Create the default list of comparison models.
    Args:
        pprogram : a probabilistic program for the covariates
        n_samples: int: the number of samples to draw
    """

    hmc_sampler = Hamiltionian_MC(pprogram=pprogram, n_warmup=n_samples//2, n_samples=n_samples)

    vi_diag = Variational_InferenceAutoguide(
        pprogram=pprogram,
        make_guide_fun=AutoDiagonalNormal,
        n_steps=2000,
        n_samples=n_samples,
        lr=1e-2,
        discrete_z=discrete_z
    )

    vi_multivariate_normal = Variational_InferenceAutoguide(
        pprogram=pprogram,
        make_guide_fun=AutoMultivariateNormal,
        n_steps=2000,
        n_samples=n_samples,
        lr=1e-2,
        discrete_z=discrete_z
    )
    vi_laplace = Variational_InferenceAutoguide(
        pprogram=pprogram,
        make_guide_fun=AutoLaplaceApproximation,
        n_steps=2000,
        n_samples=n_samples,
        lr=1e-2,
        discrete_z=discrete_z   
    )

    vi_autoIAF = Variational_InferenceAutoguide(
        pprogram=pprogram,
        make_guide_fun=AutoIAFNormal,
        n_steps=2000,
        n_samples=n_samples,
        lr=1e-3,
        discrete_z=discrete_z
    )

    vi_autostrucured = Variational_InferenceAutoguide(
        pprogram=pprogram,
        make_guide_fun=AutoStructured,
        n_steps=2000,
        n_samples=n_samples,
        lr=1e-2,
        discrete_z=discrete_z
    )

    model_list = [
        hmc_sampler,
        vi_diag,
        vi_multivariate_normal,
        vi_laplace,
        vi_autoIAF,
        vi_autostrucured
    ]

    return model_list


def make_reduced_list_comparison(
        pprogram,
        n_samples: int = 1000,
        discrete_z: bool = True
        ):
    """
    Create the reduced list of comparison models.
    Args:
        pprogram: a probabilistic program
        n_samples: int: the number of samples to draw
    """

    vi_diag = Variational_InferenceAutoguide(
        pprogram=pprogram,
        make_guide_fun=AutoDiagonalNormal,
        n_steps=2000,
        n_samples=n_samples,
        lr=1e-2,
        discrete_z=discrete_z
    )

    vi_multivariate_normal = Variational_InferenceAutoguide(
        pprogram=pprogram,
        make_guide_fun=AutoMultivariateNormal,
        n_steps=2000,
        n_samples=n_samples,
        lr=1e-2,
        discrete_z=discrete_z
    )

    model_list = [
        vi_diag,
        vi_multivariate_normal,
    ]

    return model_list


def make_hmc_numpyro_comparison(
        pprogram,
        n_samples: int = 1000,
        ):
    hmc = Hamiltionian_MC_Numpyro(pprogram=pprogram, n_samples=n_samples, n_warmup=n_samples//2)

    return [hmc]
