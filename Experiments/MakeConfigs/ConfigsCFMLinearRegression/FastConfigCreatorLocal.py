import configparser
import time 

from ModelConfigCreator import make_basic_model_config
from PFNExperiments.Experiments.MakeConfigs.ConfigsCFMLinearRegression.FastConfigCreator import FastConfigCreator

class FastConfigCreatorLocal(FastConfigCreator):
    """
    A class to create a basic configuration file for local experiments.
    """
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
            "N": 50,   # number of samples per in-context dataset
            "P": 5,    # number of features per in-context dataset
            "Batch_size": 8,  # batch size for training
            "N_epochs": 1,  # number of epochs for training
            "N_samples_per_epoch": 20, # number of samples to use per epoch
            "N_samples_to_generate_at_once": 1_000, # number of samples to generate at once
            "Shuffle": False, # shuffle the data before training
            "Save_path": r"" + f"\\{self.config_name}_{datetime}", # path to save the results
            "Train_frac": 0.5, # fraction of the data to use for training
            "Val_frac": 0.1 # fraction of the data to use for validation
        }

        self.config['BASIC']['N_batches_per_epoch'] = str(max(1, int(self.config['BASIC']['N_samples_per_epoch']) // int(self.config['BASIC']['Batch_size'])))



    def set_data_generation_params(self):
        """
        Set the data generation parameters for the configuration file.
        """

        self.config['DATA_GENERATION'] = {
            "Pprogram": "ig", # probabilistic program to generate the data
            #"Pprogram_batched": "None", # probabilistic program to generate the data in batches
            "Use_intercept": False, # whether to use an intercept in the model
            "Scheduler_behaviour": "All_constant", # behaviour of the scheduler of the probabilistic program's parameters
            "Generate_X_behaviour": "TabPFNX_extended1", # behaviour of the data generation process
            "X_data_files": [
                r""
            ],
            "pprogram_params": {
                "a": 5.0,
                "b": 2.0,
                "tau": 1.0
            } # parameters for the probabilistic program. Is a dictionary
        }


    def set_model_params(
            self
        ):
        """
        Set the model parameters for the configuration file.

        """
        P = int(self.config['BASIC']['P'])

        self.config['MODEL'] = make_basic_model_config(
            P = P,
            use_intercept = True if self.config['DATA_GENERATION']['Use_intercept'] == "True" else False,
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
            "N_samples_per_model": 200, # number of samples to generate per model
            "N_synthetic_cases": 5, # number of synthetic cases to generate and evaluate on
            "Real_world_eval": "Basic1", # real world evaluation to perform,
            "n_evaluation_cases_real_world": 3, # number of real world evaluation cases to use
            "do_full_evaluation": False, # whether to do a full evaluation
            "save_path_data_real_world_eval": r"", # path to save the data for real world evaluation
            "real_world_benchmark_id": 336 # id of the real world benchmark
        }

    def set_full_model_params(self):
        """
        Set the params for the full model.
        """

        P = int(self.config['BASIC']['P'])
        self.config['FULL_MODEL'] = {
            "sample_name": "beta",
            "sample_shape": (P+1,) if self.config['DATA_GENERATION']['Use_intercept'] == "True" else (P,),
            "n_samples": self.config["EVALUATION"]["N_samples_per_model"], # number of samples to generate to compare for each case
            "batch_size": self.config['BASIC']['Batch_size'], # batch size for generating samples
            "solve_adjoint": True, # whether to solve the adjoint ODE
            "atol": 1e-3,
            "rtol": 1e-3
        }

if __name__ == "__main__":
    config_creator = FastConfigCreatorLocal(
        config_name = "test_config_v1",
        config_path = r""
    )
    config_creator.create_config()
    config_creator.save_config()