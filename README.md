# ICL for Full Bayesian Inference

Welcome to the official code repository for our submission: "In-context Learning for Full Bayesian Inference". This repository contains the code for all experiments conducted in the paper which demonstrate that in-context learning is effective for full Bayesian inference with real-world datasets.

# Installation

To install the required dependencies, please run the following command:

```bash
pip install -r requirements.txt
```

# Experiments

## Latent Factor Models

To reproduce the experiments regarding the latent factor models, first choose a configuration file from the `Experiments/Configs/LFM_Configs` directory. You might want to adjust the `save_path' parameter to specify the directory where the results should be saved. Then, run the experiment: 

```python
from PFNExperiments.Experiments.RunExperiments.RunExperiments_LFM import RunExperiments_LFM

experiment = RunExperiments_LFM(config_path)  # initialize the experiment with the path to the configuration file
experiment.run()  # run the experiment
```

This will create a directory where the training logs and the results are saved.

## Generalized Linear Models

To reproduce the experiments regarding the generalized linear models, first choose a configuration file from the `Experiments/Configs/LM_Configs` directory. You might want to adjust the `save_path' parameter to specify the directory where the results should be saved. 

If you want to use covariates from the TabPFN prior, you first need to generate them and store them

```python
import torch
from PFNExperiments.LinearRegression.GenerativeModels.GenerateX_TabPFN.GenerateX_to_save import GenerateX_to_save

gen = GenerateX_to_save(
    n_samples = 10_000_000,
    N = 50,
    P = 5,
    save_folder= "your_save_folder",
    save_name = "X_tabpfn_n50_p5_10_000_000.pt",
    replace_nan = True,
    normalize = True
)

r = gen.sample()

torch.save(r, "your_save_folder/X_tabpfn_n50_p5_10_000_000.pt")
```

Then, add the path or several paths to the generated covariates to the configuration file under the attribute `x_data_files`.

Finally, run the experiment:

```python
from PFNExperiments.Experiments.RunExperiments.RunExperiments_LM import RunExperiments_LM

experiment = RunExperiments_LM(config_path)  # initialize the experiment with the path to the configuration file

experiment.run()  # run the experiment
```

This will create a directory where the training logs and the results are saved.

## Ablation for a Diffusion Objective using Variance Preserving Paths

To reproduce the ablation on the role of using a diffusion objective with variance preserving paths [1] compared to the standard OT objective, simply run the configs from `Experiments/Configs/Diffuion_Experiments/LM_Configs` and `Experiments/Configs/Diffuion_Experiments/LFM_Configs` directories in the same way as for the previous experiments.

## Ablation for SGLD 

To obtain results for SGLD, just choose a standard configuration file from the `Experiments/Configs/LM_Configs` or `Experiments/Configs/LFM_Configs`. Then run the file `Experiments/RunExperiments_LM_SGLD.py` for the LMs and run the file `Experiments/RunExperiments_LFM_SGLD.py` for the LFMs.

We use SGLD from [2] with preconditioning introduced in [3].

## Run the OOD experiments

To run the OOD experiments, choose a config file from `Experiments/Configs/LM_Configs_OOD` or `Experiments/Configs/FA_Configs_OOD` or `Experiments/Configs/GMM_Configs_OOD` and run `Experiments/RunExperiments_OOD.py` for the LM configs, and `Experiments/RunExperiments_LFM_ood.py` for the FA and GMM configs. 

## Run the experiments using an MLP instead of a transformer encoder

To run the experiments where an MLP encoder instead of a transformer encoder is used, choose a config file from `Experiments/Configs/LM_Configs_MLP` or `Experiments/Configs/FA_Configs_MLP`, or `Experiments/Configs/GMM_Configs_MLP` and run `Experiments/RunExperiments_LM_MLP.py` for the LM configs, and `Experiments/RunExperiments_LFM_MLP.py` for the FA and GMM configs.


# References

[1] Song, Yang, et al. "Score-based generative modeling through stochastic differential equations." arXiv preprint arXiv:2011.13456 (2020).

[2] Welling, Max, and Yee W. Teh. "Bayesian learning via stochastic gradient Langevin dynamics." Proceedings of the 28th international conference on machine learning (ICML-11). 2011.

[3] Li, Chunyuan, et al. "Preconditioned stochastic gradient Langevin dynamics for deep neural networks." Proceedings of the AAAI conference on artificial intelligence. Vol. 30. No. 1. 2016.
