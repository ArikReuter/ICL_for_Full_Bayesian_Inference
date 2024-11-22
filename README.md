# PFNExperiments

Welcome to the official code repository for our submission: "In-context Learning for Full Bayesian Inference". This repository contains the code for all experiments conducted in the paper.

# Repository Status

This repository currently serves as the code basis for our submission and is therefore maintained in an anonymized format to comply with review requirements. Upon acceptance of our work, we plan to release the complete code in a more comprehensive package format, making it publicly available for the community. 

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

## Ablation into Diffusion objective using variance preserving paths

To reproduce the ablation on the role of using a diffusion objective with variance preserving paths compared to the standard OT objective, simply run the configs from 'Experiments/Configs/Diffuion_Experiments/LM_Configs' and 'Experiments/Configs/Diffuion_Experiments/LFM_Configs' directories in the same way as the previous experiments.

