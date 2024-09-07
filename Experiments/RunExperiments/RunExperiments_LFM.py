import configparser
import ast
from copy import deepcopy
import os
import sys

from PFNExperiments.LatentFactorModels.GenerativeModels.LatenFactorModel_abstract import return_only_x
from PFNExperiments.LinearRegression.Models.Transformer_CNF import TransformerCNFConditionalDecoder
from PFNExperiments.Training.FlowMatching.CFMLossOT2 import CFMLossOT2
from PFNExperiments.LinearRegression.Models.ModelToPosteriorCNF import ModelToPosteriorCNF
from PFNExperiments.LinearRegression.GenerativeModels.Curriculum import Curriculum
from PFNExperiments.LinearRegression.GenerativeModels.GenerateDataCurriculumCFM import GenerateDataCurriculumCFM
from PFNExperiments.Training.TrainerCurriculumCNF import TrainerCurriculumCNF
from PFNExperiments.Evaluation.Evaluate import Evaluate, result_dict_to_latent_variable_convert_mu_sigma_to_beta, results_dict_to_data_x_tuple, results_dict_to_data_x_tuple_transpose, result_dict_to_latent_variable_convert_z_to_beta
from PFNExperiments.Evaluation.RealWorldEvaluation.EvaluateRealWorld import EvaluateRealWorld, just_return_results, results_dict_to_latent_variable_beta0_and_beta
from PFNExperiments.LinearRegression.GenerativeModels.GenerateX_TabPFN.MakeGenerator import MakeGenerator
from PFNExperiments.LinearRegression.GenerativeModels.GenerateX import simulate_X_uniform
from torch.optim.lr_scheduler import OneCycleLR
from PFNExperiments.LatentFactorModels.ComparisonModels.MakeDefaultListComparison import make_default_list_comparison, make_reduced_list_comparison
from PFNExperiments.Evaluation.RealWorldEvaluation.PreprocessDataset import Preprocessor
from PFNExperiments.Evaluation.RealWorldEvaluation.GetDataOpenML import GetDataOpenML
from PFNExperiments.LinearRegression.GenerativeModels.Name2Pprogram import name2pprogram_maker
from PFNExperiments.Experiments.RunExperiments.RunExperiments import RunExperiments
import torch

from PFNExperiments.Evaluation.RealWorldEvaluation.Preprocess_univariate_GMM import Preprocessor_GMM_univariate
from PFNExperiments.LatentFactorModels.Training.TrainerCurriculumCNF_LatentFactor import TrainerCurriculumCNF_LatentFactor
from PFNExperiments.LinearRegression.GenerativeModels.LM_abstract import print_code

from PFNExperiments.Evaluation.RealWorldEvaluation.Preprocess_multivariate_GMM import Preprocessor_GMM_multivariate

from PFNExperiments.Training.Trainer import visualize_training_results
#from PFNExperiments.Evaluation.RealWorldEvaluation.PreprocessName2Preprocessor import name2preprocessor

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

class RunExperiments_LFM(RunExperiments):
    """
    A class to run experiments in Colab.
    """
        
    def __init__(
                self,
                config_path: str = "./config_lfm.ini",
        ):
        """
        Constructor of the RunExperimentsColab class.
        Args:
        config_path: str: the path to save the configuration file
        """
        self.config_path = config_path
        self.config = configparser.ConfigParser() # Create a ConfigParser object
        
        self.config.read(f"{self.config_path}")

        # save config at the save path
        # if target forlder does not exist, create it
        if not os.path.exists(self.config["BASIC"]["Save_path"]):
            os.makedirs(self.config["BASIC"]["Save_path"])

        with open(f"{self.config['BASIC']['Save_path']}/config.ini", "w") as configfile:
            self.config.write(configfile)


    def setup_data_generation(self):
        """
        Setup the data generation.
        """
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

        if gen_config["Generate_X_behaviour"] == "uniform":
            self.generate_X = simulate_X_uniform

        pprogram_batched, pprogram = name2pprogram_maker[gen_config["pprogram"]]

        self.data_generator = GenerateDataCurriculumCFM(
            pprogram_maker = pprogram_batched,
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


    def setup_training(self):
        """
        Setup the training.
        """
        
        train_config = self.config["TRAINING"]

        if train_config["Loss_function"] == "CFMLossOT2":
            self.loss_function = CFMLossOT2(
                sigma_min = float(train_config["Sigma_min"])
            )
        else:
            raise ValueError(f"Loss function {train_config['Loss_function']} not implemented yet!")
        
        self.optimizer = torch.optim.Adam(
                                          self.model.parameters(), 
                                          lr=float(self.config["TRAINING"]["Learning_rate"]),
                                          weight_decay=float(self.config["TRAINING"]["Weight_decay"]))
        
        scheduler_params = ast.literal_eval(train_config["Scheduler_params"])
        self.scheduler = OneCycleLR(
            optimizer = self.optimizer,
            max_lr = float(scheduler_params["max_lr"]),
            epochs = int(scheduler_params["epochs"]),
            steps_per_epoch = int(scheduler_params["steps_per_epoch"]),
            pct_start = float(scheduler_params["pct_start"]),
            div_factor = float(scheduler_params["div_factor"]),
            final_div_factor = float(scheduler_params["final_div_factor"])

        )

        self.trainer = TrainerCurriculumCNF_LatentFactor(
            model = self.model,
            optimizer = self.optimizer,
            scheduler = self.scheduler,
            loss_function = self.loss_function,
            epoch_loader = self.epoch_loader,
            evaluation_functions = {},
            n_epochs = int(self.config["BASIC"]["N_epochs"]),
            early_stopping_patience = int(self.config["TRAINING"]["Early_stopping_patience"]),
            schedule_step_on = "batch",
            save_path = self.config["BASIC"]["Save_path"],
            coupling = None,
            use_same_timestep_per_batch = False,
            use_train_mode_during_validation = False,
            max_gradient_norm = float(self.config["TRAINING"]["max_grad_norm"]),

        )

        initial_val_loss = self.trainer.validate()

        print(f"Initial validation loss: {initial_val_loss}")

        r = self.trainer.train()

        visualize_training_results(r, loglog = False)

        self.trainer.load_best_model()
        self.model = self.trainer.model
        self.model.eval()

        test_res = self.trainer.test()

        print(f"Test results: {test_res}")

    def load_model_from_path(self, new_save_path: str, validate = True):
        """
        Load the model from a path.
        Args:
        new_save_path: str: the path to load the model from
        validate: bool: whether to validate the model 
        """

        train_config = self.config["TRAINING"]

        if train_config["Loss_function"] == "CFMLossOT2":
            self.loss_function = CFMLossOT2(
                sigma_min = float(train_config["Sigma_min"])
            )
        else:
            raise ValueError(f"Loss function {train_config['Loss_function']} not implemented yet!")
        
        self.optimizer = torch.optim.Adam(
                                          self.model.parameters(), 
                                          lr=float(self.config["TRAINING"]["Learning_rate"]),
                                          weight_decay=float(self.config["TRAINING"]["Weight_decay"]))
        
        scheduler_params = ast.literal_eval(train_config["Scheduler_params"])
        self.scheduler = OneCycleLR(
            optimizer = self.optimizer,
            max_lr = float(scheduler_params["max_lr"]),
            epochs = int(scheduler_params["epochs"]),
            steps_per_epoch = int(scheduler_params["steps_per_epoch"]),
            pct_start = float(scheduler_params["pct_start"]),
            div_factor = float(scheduler_params["div_factor"]),
            final_div_factor = float(scheduler_params["final_div_factor"])

        )

        self.trainer = TrainerCurriculumCNF_LatentFactor(
            model = self.model,
            optimizer = self.optimizer,
            scheduler = self.scheduler,
            loss_function = self.loss_function,
            epoch_loader = self.epoch_loader,
            evaluation_functions = {},
            n_epochs = int(self.config["BASIC"]["N_epochs"]),
            early_stopping_patience = int(self.config["TRAINING"]["Early_stopping_patience"]),
            schedule_step_on = "batch",
            save_path = self.config["BASIC"]["Save_path"],
            coupling = None,
            use_same_timestep_per_batch = False,
            use_train_mode_during_validation = False,
            max_gradient_norm = float(self.config["TRAINING"]["max_grad_norm"]),

        )
        self.trainer.set_new_save_path(new_save_path)

        self.trainer.load_best_model()
        self.model = self.trainer.model
        self.model.eval()

        if validate:
            initial_val_loss = self.trainer.validate()

            print(f"Initial validation loss: {initial_val_loss}")

    def setup_full_model(self):
        """
        Setup the full model.
        """
        full_model_kwargs = self.config["FULL_MODEL"]

        self.full_model = ModelToPosteriorCNF(
            model = self.model,
            sample_shape= ast.literal_eval(full_model_kwargs["sample_shape"]),
            sample_name= full_model_kwargs["sample_name"],
            n_samples= int(full_model_kwargs["n_samples"]),
            batch_size= int(full_model_kwargs["batch_size"]),
            solve_adjoint= string2bool(full_model_kwargs["solve_adjoint"]),
            atol = float(full_model_kwargs["atol"]),
            rtol = float(full_model_kwargs["rtol"]),
        )

    def setup_evaluation(self):
        """
        Setup the evaluation.
        """

        benchmark_params_ppgrogram = self.data_generator.curriculum.get_params(-1)

        del benchmark_params_ppgrogram["batch_size"]

        print(f"params for pprogram: {benchmark_params_ppgrogram}")

        self.pprogram1 = name2pprogram_maker[self.config["DATA_GENERATION"]["Pprogram"]][1](**benchmark_params_ppgrogram)

        assert self.pprogram1 is not None, "pprogram1 is None!"

        self.pprogram1_y = return_only_x(self.pprogram1)
        
        print_code(self.pprogram1)

        if "discrete_z" in self.config["EVALUATION"].keys():
            discrete_z = string2bool(self.config["EVALUATION"]["discrete_z"])
        else:
            discrete_z = True

        if string2bool(self.config["EVALUATION"]["do_full_evaluation"]) is True:
            self.comparison_models = make_default_list_comparison(
                pprogram=self.pprogram1_y,
                n_samples=int(self.config["EVALUATION"]["N_samples_per_model"]),
                discrete_z = discrete_z
            )
        else:
            self.comparison_models = make_reduced_list_comparison(
                pprogram_y=self.pprogram1_y,
                n_samples=int(self.config["EVALUATION"]["N_samples_per_model"]),
                discrete_z = discrete_z
            )

    def evaluate_synthetic(self):
        """
        Evaluate the synthetic data.
        """

        if self.config["EVALUATION"]["result_dict_to_data_for_comparison_models"] == "results_dict_to_data_x_tuple_transpose":
            results_dict_to_data_x = results_dict_to_data_x_tuple_transpose
        else:
            results_dict_to_data_x = results_dict_to_data_x_tuple

        if "results_dict_to_latent_variable_comparison_models" in self.config["EVALUATION"].keys():
            if self.config["EVALUATION"]["results_dict_to_latent_variable_comparison_models"] == "result_dict_to_latent_variable_convert_z_to_beta":
                result_dict_to_latent_variable_comparison = result_dict_to_latent_variable_convert_z_to_beta
            else:
                result_dict_to_latent_variable_comparison = result_dict_to_latent_variable_convert_mu_sigma_to_beta

        self.evaluator = Evaluate(
        posterior_model=self.full_model,
        evaluation_loader=self.trainer.testset,
        comparison_models=self.comparison_models,
        n_evaluation_cases = int(self.config["EVALUATION"]["N_synthetic_cases"]),
        save_path = self.config["BASIC"]["Save_path"] + "/synthetic_evaluation",
        results_dict_to_data_for_model = results_dict_to_data_x_tuple,
        results_dict_to_latent_variable_comparison_models= result_dict_to_latent_variable_comparison,
        result_dict_to_data_for_comparison_models =  results_dict_to_data_x,
        overwrite_results=True
        )

        self.eval_res_synthetic = self.evaluator.run_evaluation()
        self.evaluator.plot_results(max_number_plots=int(self.config["EVALUATION"]["N_synthetic_cases"]))

    def evaluate_real_world(self):
        """
        Evaluate the real world data.
        """
        target_mean = self.check_model_res[1]["X"]['mean_mean']
        target_var = self.check_model_res[1]["X"]['variance_mean']

        if self.config["EVALUATION"]["result_dict_to_data_for_comparison_models"] == "results_dict_to_data_x_tuple_transpose":
            results_dict_to_data_x = results_dict_to_data_x_tuple_transpose
        else:
            results_dict_to_data_x = results_dict_to_data_x_tuple

        if "results_dict_to_latent_variable_comparison_models" in self.config["EVALUATION"].keys():
            if self.config["EVALUATION"]["results_dict_to_latent_variable_comparison_models"] == "result_dict_to_latent_variable_convert_z_to_beta":
                result_dict_to_latent_variable_comparison = result_dict_to_latent_variable_convert_z_to_beta
            else:
                result_dict_to_latent_variable_comparison = result_dict_to_latent_variable_convert_mu_sigma_to_beta

        if self.config["EVALUATION"]["real_world_preprocessor"] == "gmm_preprocessor_univariate":
            self.getdata = GetDataOpenML(
                preprocessor = Preprocessor_GMM_univariate(
                    N_datapoints = int(self.config["BASIC"]["N"]),
                    x_mean = target_mean,
                    x_var = target_var,
                ),
                save_path = self.config["EVALUATION"]["save_path_data_real_world_eval"],
                benchmark_id = self.config["EVALUATION"]["real_world_benchmark_id"]
            )

        if self.config["EVALUATION"]["real_world_preprocessor"] == "gmm_preprocessor_multivariate":
            self.getdata = GetDataOpenML(
                preprocessor = Preprocessor_GMM_multivariate(
                    N_datapoints = int(self.config["BASIC"]["N"]),
                    P_features = int(self.config["BASIC"]["P"]),
                    x_mean = target_mean,
                    x_var = target_var,
                ),
                save_path = self.config["EVALUATION"]["save_path_data_real_world_eval"],
                benchmark_id = self.config["EVALUATION"]["real_world_benchmark_id"]
            )
            
        self.datasets = self.getdata.get_data()

        if self.config["EVALUATION"]["n_evaluation_cases_real_world"] == "All":
            n_evaluation_cases = len(self.datasets)
        else:
            n_evaluation_cases = int(self.config["EVALUATION"]["n_evaluation_cases_real_world"])

        self.eval_rw = EvaluateRealWorld(
            posterior_model = self.full_model,
            evaluation_datasets = self.datasets,
            comparison_models = self.comparison_models,
            results_dict_to_data_for_model = results_dict_to_data_x_tuple,
            results_dict_to_latent_variable_comparison_models= result_dict_to_latent_variable_comparison, 
            result_dict_to_data_for_comparison_models = results_dict_to_data_x,           
            n_evaluation_cases = n_evaluation_cases,
            save_path = self.config["BASIC"]["Save_path"] + "/real_world_evaluation",
            overwrite_results = True
        )

        self.eval_res_real_world = self.eval_rw.run_evaluation()

        self.eval_rw.plot_results()

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


        
        