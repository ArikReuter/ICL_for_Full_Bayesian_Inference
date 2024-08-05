
from datetime import datetime

import torch

import os
from tqdm import tqdm

#from tabpfn.scripts.differentiable_pfn_evaluation import eval_model_range
from tabpfn.scripts.model_builder import get_model


from tabpfn.scripts.model_configs import *  # noqa: F403


class GenerateX_to_save():

    """
    use the tabpfn prior to simulate data and save it to disk
    """
    def __init__(
        self,
        n_samples: int = 100_000,
        N: int = 50,
        P: int = 5,
        save_folder: str = 'data/',
        save_name: str = 'data_x_tabpfn.pt',
        replace_nan: bool = True
    ):
        """
        Args:
            n_samples: int: the number of samples
            N: int: the number of observations
            P: int: the number of covariates
            save_path: str: the path to save the data
            replace_nan: bool: whether to replace nan values with 0
        """
        self.n_samples = n_samples
        self.N = N
        self.P = P
        
        self.save_path = os.path.join(save_folder, save_name)

        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
          


    def sample(self) -> torch.Tensor:
        """
        sample the data
        """

        device = 'cpu'

        def reload_config(config_type='causal', task_type='multiclass', longer=0):
            config = get_prior_config(config_type=config_type)
            config['prior_type'], config['differentiable'], config['flexible'] = 'mlp', True, True
            model_string = ''
            config['epochs'] = 12000
            config['recompute_attn'] = True
            config['max_num_classes'] = 10
            config['num_classes'] = uniform_int_sampler_f(2, config['max_num_classes'])
            config['balanced'] = False
            model_string = model_string + '_multiclass'
            model_string = model_string + '_'+datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
            
            return config, model_string
        

        config, model_string = reload_config(longer=1)
        config['bptt_extra_samples'] = None

        # diff
        config['output_multiclass_ordered_p'] = 0.
        del config['differentiable_hyperparameters']['output_multiclass_ordered_p']
        config['multiclass_type'] = 'rank'
        del config['differentiable_hyperparameters']['multiclass_type']
        config['sampling'] = 'normal' # vielleicht schlecht?
        del config['differentiable_hyperparameters']['sampling']
        config['pre_sample_causes'] = True
        # end diff
        config['multiclass_loss_type'] = 'nono' # 'compatible'
        config['normalize_to_ranking'] = False # False
        config['categorical_feature_p'] = .2 # diff: .0
        # turn this back on in a random search!?
        config['nan_prob_no_reason'] = .0
        config['nan_prob_unknown_reason'] = .0 # diff: .0
        config['set_value_to_nan'] = .1 # diff: 1.
        config['normalize_with_sqrt'] = False
        config['new_mlp_per_example'] = True
        config['prior_mlp_scale_weights_sqrt'] = True
        config['batch_size_per_gp_sample'] = None
        config['normalize_ignore_label_too'] = False
        config['differentiable_hps_as_style'] = False
        config['max_eval_pos'] = 1000
        config['random_feature_rotation'] = True
        config['rotate_normalized_labels'] = True
        config["mix_activations"] = False # False heisst eig True
        config['emsize'] = 512
        config['nhead'] = config['emsize'] // 128
        config['bptt'] = self.N
        config['canonical_y_encoder'] = False  
        config['aggregate_k_gradients'] = 8
        config['batch_size'] = 8*config['aggregate_k_gradients']
        config['num_steps'] = int(1e10)
        config['epochs'] = int(1e10)
        config['total_available_time_in_s'] = None #60*60*22 # 22 hours for some safety...
        config['train_mixed_precision'] = True
        config['efficient_eval_masking'] = True
        config_sample = evaluate_hypers(config)
        config_sample["num_features_used"] = {'uniform_int_sampler_f(3,max_features)': uniform_int_sampler_f(self.P, self.P)}
        config_sample["num_features"] = self.P

        config_sample["batch_size"] = 4

        model = get_model(config_sample, device, should_train=False, verbose=-10) # , state_dict=model[2].state_dict()

        data_res = []
        for i in tqdm(list(range(self.n_samples))):
            (hp_embedding, data, _), targets, single_eval_pos = next(iter(model[3]))
            data_res.append(data.squeeze().cpu())

        data_res = torch.stack(data_res)

        if self.replace_nan:
            data_res[torch.isnan(data_res)] = 0

        torch.save(data_res, self.save_path)

        return data_res
    

