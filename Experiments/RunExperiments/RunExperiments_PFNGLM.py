import configparser
import ast
from copy import deepcopy
import os
import sys

from PFNExperiments.LinearRegression.GenerativeModels.LM_abstract import return_only_y, print_code
from PFNExperiments.LinearRegression.Models.Transformer_CNF import TransformerCNFConditionalDecoder, TransformerCNFConditionalDecoderSequenceZ
from PFNExperiments.Training.FlowMatching.CFMLossOT2 import CFMLossOT2
from PFNExperiments.LinearRegression.Models.ModelToPosteriorCNF import ModelToPosteriorCNF
from PFNExperiments.LinearRegression.GenerativeModels.Curriculum import Curriculum
from PFNExperiments.LinearRegression.GenerativeModels.GenerateDataCurriculumCFM import GenerateDataCurriculumCFM
from PFNExperiments.Training.TrainerCurriculumCNF import TrainerCurriculumCNF
from PFNExperiments.Evaluation.Evaluate import Evaluate
from PFNExperiments.Evaluation.RealWorldEvaluation.EvaluateRealWorld import EvaluateRealWorld, just_return_results, results_dict_to_latent_variable_beta0_and_beta
from PFNExperiments.LinearRegression.GenerativeModels.GenerateX_TabPFN.MakeGenerator import MakeGenerator
from torch.optim.lr_scheduler import OneCycleLR
from PFNExperiments.LinearRegression.ComparisonModels.MakeDefaultListComparison import make_default_list_comparison, make_reduced_list_comparison
from PFNExperiments.Evaluation.RealWorldEvaluation.PreprocessDataset import Preprocessor
from PFNExperiments.Evaluation.RealWorldEvaluation.GetDataOpenML import GetDataOpenML
from PFNExperiments.LinearRegression.GenerativeModels.Name2Pprogram import name2pprogram_maker
from PFNExperiments.Training.Trainer import visualize_training_results
import torch

from PFNExperiments.Experiments.RunExperiments.RunExperiments import RunExperiments

def string2bool(s: str) -> bool:
    """
    Convert a string to a boolean.
    Args:
    s: str: the string to convert
    Returns:
    bool: the boolean value
    """
    if s.lower() in ["true", "1"]:
        return True
    elif s.lower() in ["false", "0"]:
        return False
    else:
        raise ValueError(f"String {s} not recognized as boolean!")

class RunExperiments_PFNGLM(RunExperiments):
    """
    A class to run experiments in Colab.
    """
        
    def __init__(
                self,
                config_path: str = "./config.ini",
        ):
        """
        Constructor of the RunExperimentsColab class.
        Args:
        config_path: str: the path to save the configuration file
        """
        super().__init__(config_path)

    
    def setup_data_generation(self):
        """
        Setup the data generation."""
        gen_config = self.config["DATA_GENERATION"]
        N_EPOCHS = int(self.config["BASIC"]["N_epochs"])
        N_BATCHES_PER_EPOCH = int(self.config["BASIC"]["N_batches_per_epoch"])
        BATCH_SIZE = int(self.config["BASIC"]["Batch_size"])
        P = int(self.config["BASIC"]["P"])
        N = int(self.config["BASIC"]["N"])
        N_SAMPLES_PER_EPOCH = int(self.config["BASIC"]["N_samples_per_epoch"])

        pprogram_params = ast.literal_eval(gen_config["pprogram_params"])

        self.curriculum = Curriculum(max_iter=int(N_EPOCHS*N_BATCHES_PER_EPOCH*BATCH_SIZE*0.5))
        param_list = [(name, self.curriculum.constant_scheduler(float(value))) for name, value in pprogram_params.items()]

        self.curriculum.add_param_list(
            param_list
        )
        self.curriculum.plot_all_schedules()

        if gen_config["Generate_X_behaviour"] == "TabPFNX_extended1":
            X_files_paths_str = gen_config["X_data_files"]
            # parse a string of paths to a list of paths
            X_files_paths = ast.literal_eval(X_files_paths_str)
            self.generate_X = MakeGenerator(
                paths = X_files_paths
            ).make_generator()

            X_data =  MakeGenerator(
                paths = X_files_paths
            ).load_data()  # only load the data to obtain X_tabpfn

        pprogram_maker_class = name2pprogram_maker[gen_config["pprogram"]]  # load a pprogram_maker_class

        pprogram_maker = pprogram_maker_class(X_data)  # this class takes the data as input and yields a function that generates probabilistic programs

        self.data_generator = GenerateDataCurriculumCFM(
            pprogram_maker = pprogram_maker,
            curriculum= self.curriculum,
            pprogram_covariates = self.generate_X,
        )

        self.check_model_res = self.data_generator.check_model(
            n_samples_per_epoch=N_SAMPLES_PER_EPOCH,
            epochs_to_check = [0, N_EPOCHS-1],
            p = P,
            n = N,
            used_batch_samples = 3,
            save_path_plots=self.config["BASIC"]["Save_path"] + "/check_model"
        )
        self.epoch_loader = self.data_generator.make_epoch_loader(
            n = N,
            p = P,
            number_of_batches_per_epoch = N_BATCHES_PER_EPOCH,
            n_epochs = N_EPOCHS,
            batch_size= BATCH_SIZE,
            train_frac= float(self.config["BASIC"]["Train_frac"]),
            val_frac= float(self.config["BASIC"]["Val_frac"]),
            shuffle= string2bool(self.config["BASIC"]["Shuffle"]),
            n_samples_to_generate_at_once = int(self.config["BASIC"]["N_samples_to_generate_at_once"]),
        )

        sample_batch = next(iter(self.epoch_loader[0][0])) # try if loader works


    def setup_evaluation(self):
        pass
        
    def evaluate_synthetic(self):
        pass
    def evaluate_real_world(self):
       pass

    def run(self):
        """
        Run the experiments.
        """
        stdout = sys.stdout
        log_file = open(self.config["BASIC"]["Save_path"] + "/log.txt", "w")
        sys.stdout = log_file
        sys.stderr = log_file

        try:
            print("Starting experiments")

            self.setup_data_generation()
            self.setup_model()
            self.setup_training()
            self.setup_full_model()
            self.setup_evaluation()
            self.evaluate_synthetic()
            self.evaluate_real_world()

            print("Experiments finished")
        finally:
            log_file.flush()
            os.fsync(log_file.fileno())
            log_file.close()
            sys.stdout = stdout
            sys.stderr = stdout


        
        