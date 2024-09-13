from PFNExperiments.LatentFactorModels.GenerativeModels.FactorAnalysis.BasicFA import make_fa_program_normal_weight_prior as make_fa_program_normal_weight_prior_pyro
from PFNExperiments.LatentFactorModels.GenerativeModels.Numpyro_Versions.BasicFA import make_fa_program_normal_weight_prior as make_fa_program_normal_weight_prior_numpyro

"""
Simply map the Pyro code to Numpyro code
"""

pyro_ppgram2_numpyro_ppgram = {
    str(make_fa_program_normal_weight_prior_pyro.__name__): make_fa_program_normal_weight_prior_numpyro
}
