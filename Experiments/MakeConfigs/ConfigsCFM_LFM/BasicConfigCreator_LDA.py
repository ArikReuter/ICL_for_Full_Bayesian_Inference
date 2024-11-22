import configparser
import time 
import ast

from PFNExperiments.Experiments.MakeConfigs.ConfigsCFM_LFM.ModelConfigCreator_LDA import make_basic_model_config

class BasicConfigCreator():
    """
    A class to create a basic configuration file for experiments in Colab.
    """
    
    def __init__(
            self,
            config_name: str = None,
            config_path: str = "./",
    ):
        """
        Constructor of the BasicConfigCreator class.
        Args:
        config_name: str: the name of the configuration file
        config_path: str: the path to save the configuration file
        """
        self.config = configparser.ConfigParser()

        self.config["confic_created_on"] = {"time": {time.asctime()}}

        if config_name is None:
            self.config_name = "basic_config_lda_small_eval"
        else:
            self.config_name = config_name

        self.config["name"] = {"name": self.config_name}
        self.config_path = config_path

    def create_config(self):
        """
        Create the configuration file.
        """

        self.set_basic_params()
        self.set_data_generation_params()
        self.set_model_params()
        self.set_training_params()
        self.set_evaluation_params()
        self.set_full_model_params()

        
    def save_config(self):
        """
        Save the configuration file.
        """

        with open(f"{self.config_path}/{self.config_name}.ini", "w") as configfile:
            self.config.write(configfile)

        
    def set_basic_params(self):
        """
        Set the basic parameters for the configuration file.
        """
        datetime = time.asctime()
        datetime = datetime.replace(" ", "_")
        datetime = datetime.replace(":", "_")


        self.config['BASIC'] = {
            "N" : 50,
            "P" : 100,
            "Batch_size": 512,  # batch size for training
            "N_epochs": 150,  # number of epochs for training
            "N_samples_per_epoch": 500_000, # number of samples to use per epoch
            "N_samples_to_generate_at_once": 1_000, # number of samples to generate at once
            "Shuffle": False, # shuffle the data before training
            "Save_path": "/content/drive/MyDrive/PFN_Experiments/Training_RunsCFM" + f"/{self.config_name}_{datetime}",
            "Train_frac": 0.5, # fraction of the data to use for training
            "Val_frac": 0.1 # fraction of the data to use for validation
        }

        self.config['BASIC']['N_batches_per_epoch'] = str(max(1, int(self.config['BASIC']['N_samples_per_epoch']) // int(self.config['BASIC']['Batch_size'])))



    def set_data_generation_params(self):
        """
        Set the data generation parameters for the configuration file.
        """

        self.config['DATA_GENERATION'] = {
            "Pprogram": "lda_basic", # probabilistic program to generate the data
            #"Pprogram_batched": None, # probabilistic program to generate the data in batches
            "Scheduler_behaviour": "All_constant", # behaviour of the scheduler of the probabilistic program's parameters
            "Generate_X_behaviour": "uniform", # behaviour of the data generation process
            "pprogram_params": {
                "n_docs": self.config['BASIC']['N'], # number of data points
                "n_words": self.config['BASIC']['P'], # number of features
                "batch_size": self.config['BASIC']['Batch_size'], # batch size
                "n_topics": 3, 
                "alpha_dir": 0.5,
                "beta_dir": 0.5,
                "doc_len_max": 100,
                "doc_len_mean": 10.0,
            } # parameters for the probabilistic program. Is a dictionary
        }


    def set_model_params(
            self
        ):
        """
        Set the model parameters for the configuration file.

        """
        P = int(self.config['BASIC']['P'])
        K = int(ast.literal_eval(self.config['DATA_GENERATION']['pprogram_params'])['n_topics'])

        self.config['MODEL'] = make_basic_model_config(
            P = P,
            K = K
        )

    def set_training_params(
            self
    ):
        """
        Set the training parameters for the configuration file.
        """

        self.config['TRAINING'] = {
            "Loss_function": "CFMLossOT2", # loss function to use for training
            "Sigma_min": 1e-4, # minimum value for sigma
            "Learning_rate": 1e-6, # learning rate for training
            "Weight_decay": 1e-5, # weight decay for training
            "Scheduler": "OneCycleLR", # scheduler for the learning rate
            "Scheduler_params": {
                "max_lr": 5e-4, # maximum learning rate
                "epochs": self.config['BASIC']['N_epochs'], # number of epochs
                "steps_per_epoch": max(1, int(self.config['BASIC']['N_samples_per_epoch']) // int(self.config['BASIC']['Batch_size'])), # steps per epoch
                "pct_start": 0.1, # percentage of the cycle spent increasing the learning rate
                "div_factor": 25.0, # factor by which the maximum learning rate is divided
                "final_div_factor": 1e4 # factor by which the maximum learning rate is divided at the end
            },
            "early_stopping_patience": 100_000, # patience for early stopping -> set to a large number to disable early stopping
            "max_grad_norm": 1.0, # maximum gradient norm
        }

    def set_evaluation_params(self):
        """
        Set the evaluation parameters for the configuration file.
        """

        self.config["EVALUATION"] = {
            "N_samples_per_model": 1_000, # number of samples to generate per model
            "N_synthetic_cases": 50, # number of synthetic cases to generate and evaluate on
            "Real_world_eval": "Basic1", # real world evaluation to perform,
            "n_evaluation_cases_real_world": "All", # number of real world evaluation cases to use
            "do_full_evaluation": True, # whether to do a full evaluation
            "only_use_hmc_numpyro": True, # whether to only use hmc_numpyro for the real world evaluation
            "save_path_data_real_world_eval": "/", # path to save the data for real world evaluation
            "real_world_benchmark_id": 336, # id of the real world benchmark,
            "real_world_preprocessor": "gmm_preprocessor_multivariate", # preprocessor for the real world data
            "results_dict_to_latent_variable_posterior_model" : "just_return_results_flatten_beta",
            "results_dict_to_latent_variable_comparison_models": "result_dict_to_latent_variable_convert_phi_to_beta_flatten",
            "result_dict_to_data_for_comparison_models" : "results_dict_to_data_x_tuple",
            "discrete_z": False,
        }

    def set_full_model_params(self):
        """
        Set the params for the full model.
        """

        K = int(ast.literal_eval(self.config['DATA_GENERATION']['pprogram_params'])['n_topics'])
        P = int(self.config['BASIC']['P'])

        self.config['FULL_MODEL'] = {
            "sample_name": "beta",
            "sample_shape": (K,P),
            "n_samples": self.config["EVALUATION"]["N_samples_per_model"], # number of samples to generate to compare for each case
            "batch_size": self.config['BASIC']['Batch_size'], # batch size for generating samples
            "solve_adjoint": True, # whether to solve the adjoint ODE
            "atol": 1e-7,
            "rtol": 1e-7
        }

if __name__ == "__main__":
    config_creator = BasicConfigCreator(
        config_name = "basic_config_lda_small_eval",
        config_path = r"C:\Users\arik_\Documents\Dokumente\Job_Clausthal\PFNs\Repository\PFNExperiments\Experiments\Configs\GMM_Configs"
    )
    config_creator.create_config()
    config_creator.save_config()