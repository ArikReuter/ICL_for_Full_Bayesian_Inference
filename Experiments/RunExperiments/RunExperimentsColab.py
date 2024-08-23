import configparser

from PFNExperiments.LinearRegression.GenerativeModels.GenerateData import GenerateData, check_data, check_and_plot_data
from PFNExperiments.LinearRegression.GenerativeModels.GenerateDataLM_TargetDist import make_lm_program_ig_gamma_response_batched, make_lm_program_ig_gamma_response
from PFNExperiments.LinearRegression.GenerativeModels.GenerateX import simulate_X_uniform
from PFNExperiments.LinearRegression.GenerativeModels.LM_abstract import return_only_y, print_code
from PFNExperiments.LinearRegression.GenerativeModels.Quantizer import Quantizer
from PFNExperiments.LinearRegression.Models.Transformer_CNF import TransformerCNFConditionalDecoder
from PFNExperiments.Training.Trainer import Trainer, batch_to_model_lm, visualize_training_results
from PFNExperiments.Training.Losses import MSELoss_unsqueezed, nll_loss_full_gaussian
from PFNExperiments.Training.EvalMetrics import mean_squared_error_torch_avg, mae_torch_avg, r2_score_torch_avg
from PFNExperiments.LinearRegression.Models.ModelPosterior import ModelPosteriorFullGaussian, ModelPosteriorFullGaussian2
from PFNExperiments.LinearRegression.Models.AugmentLoss import add_l2_loss_nll_loss
from PFNExperiments.LinearRegression.Models.ModelToPosteriorCNF_LearnedBaseDist import ModelToPosteriorCNF_LearnedBaseDist
from PFNExperiments.Training.FlowMatching.CFMLossOT2 import CFMLossOT2
from PFNExperiments.Evaluation.ComparePosteriorSamples import compare_all_metrics, marginal_plots_hist_parallel, marginal_plots_kde_together
from PFNExperiments.LinearRegression.Evaluation.CompareModels import ModelComparison
from PFNExperiments.LinearRegression.Models.ModelToPosteriorCNF import ModelToPosteriorCNF
from PFNExperiments.LinearRegression.ComparisonModels.Hamiltionion_MC import Hamiltionian_MC
from PFNExperiments.LinearRegression.ComparisonModels.Variational_Inference import Variational_Inference, make_guide_program_gamma_gamma
from PFNExperiments.LinearRegression.Evaluation.CompareComparisonModels import CompareComparisonModels
from PFNExperiments.LinearRegression.GenerativeModels.Curriculum import Curriculum
from PFNExperiments.LinearRegression.GenerativeModels.GenerateDataCurriculumCFM import GenerateDataCurriculumCFM
from PFNExperiments.Training.TrainerCurriculumCNF import TrainerCurriculumCNF
from PFNExperiments.LinearRegression.ComparisonModels.AnalyticalSolutionsLM import PosteriorLM_IG
from PFNExperiments.LinearRegression.GenerativeModels.GenerateDataLM_Examples import make_lm_program_ig_batched, make_lm_program_ig
from PFNExperiments.LinearRegression.GenerativeModels.GenerateX import simulate_X_uniform_discretized3
from PFNExperiments.LinearRegression.Evaluation.CompareComparisonModels import CompareComparisonModels
from PFNExperiments.Training.FlowMatching.Couplings import MiniBatchOTCoupling
from PFNExperiments.LinearRegression.ComparisonModels.Variational_InferenceAutoguide import Variational_InferenceAutoguide
from pyro.infer.autoguide import AutoDiagonalNormal, AutoMultivariateNormal, AutoLaplaceApproximation, AutoIAFNormal, AutoStructured
from PFNExperiments.Evaluation.Evaluate import Evaluate
from PFNExperiments.LinearRegression.GenerativeModels.GenerateX import make_simulate_X_by_loading
from PFNExperiments.Evaluation.RealWorldEvaluation.EvaluateRealWorld import EvaluateRealWorld, just_return_results, results_dict_to_latent_variable_beta, results_dict_to_latent_variable_beta0_and_beta
from PFNExperiments.LinearRegression.GenerativeModels.GenerateX_TabPFN import MakeGeneratorColab
from torch.optim.lr_scheduler import OneCycleLR
from PFNExperiments.LinearRegression.ComparisonModels.MakeDefaultListComparison import make_default_list_comparison, make_reduced_list_comparison
from PFNExperiments.Evaluation.RealWorldEvaluation.PreprocessDataset import Preprocessor
from PFNExperiments.Evaluation.RealWorldEvaluation.GetDataOpenML import GetDataOpenML
import torch

class RunExperimentsColab():
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
        self.config_path = config_path
        self.config = configparser.ConfigParser() # Create a ConfigParser object
        
        self.config.read(f"{self.config_path}")

    def setup_data_generation(self):
        """
        Setup the data generation.
        """
        gen_config = self.config["DATA_GENERATION"]
        N_EPOCHS = self.config["BASCIC"]["N_epochs"]
        N_BATCHES_PER_EPOCH = self.config["BASCIC"]["N_batches_per_epoch"]
        BATCH_SIZE = self.config["BASCIC"]["Batch_size"]
        P = self.config["BASCIC"]["P"]
        N = self.config["BASCIC"]["N"]
        N_SAMPLES_PER_EPOCH = self.config["BASCIC"]["N_samples_per_epoch"]

        self.curriculum = Curriculum(max_iter=int(N_EPOCHS*N_BATCHES_PER_EPOCH*BATCH_SIZE*0.5))
        self.curriculum.plot_all_schedules()

        param_list = [(name, self.curriculum.constant_scheduler(value)) for name, value in gen_config["pprogram_params"].items()]

        self.curriculum.add_param_list(
            param_list
        )
        self.curriculum.plot_all_schedules()

        self.generate_X = MakeGeneratorColab()

        self.data_generator = GenerateDataCurriculumCFM(
            pprogram_maker = self.config["DATA_GENERATION"]["Pprogram_batched"],
            curriculum= self.curriculum,
            pprogram_covariates = self.generate_X,
        )

        self.check_model_res = self.data_generator.check_model(
            n_samples_per_epoch=N_SAMPLES_PER_EPOCH,
            epochs_to_check = [0, N_EPOCHS-1],
            p = P,
            n = N,
            used_batch_samples = 100
        )

        self.epoch_loader = self.data_generator.make_epoch_loader(
            n = N,
            p = P,
            number_of_batches_per_epoch = N_BATCHES_PER_EPOCH,
            n_epochs = N_EPOCHS,
            batch_size= BATCH_SIZE,
            train_frac= self.config["BASIC"]["Train_frac"],
            val_frac= self.config["BASIC"]["Val_frac"],
            shuffle= self.config["BASIC"]["Shuffle"],
            n_samples_to_generate_at_once = self.config["BASIC"]["N_samples_to_generate_at_once"],
        )

        sample_batch = next(iter(self.epoch_loader[0][0])) # try if loader works

    def setup_model(self):
        """
        Setup the model.
        """
        
        self.model = TransformerCNFConditionalDecoder(
            **self.config["MODEL"]
        )


    def setup_training(self):
        """
        Setup the training.
        """
        
        train_config = self.config["Training"]

        if train_config["Loss_function"] == "CFMLossOT2":
            self.loss_function = CFMLossOT2(
                sigma_min=  train_config["Sigma_min"]
            )
        else:
            raise ValueError(f"Loss function {train_config["Loss_function"]} not implemented yet!")
        
        self.scheduler = OneCycleLR(
            **train_config["Scheduler_params"]
        )

        self.optimizer = torch.optim.Adam(
                                          self.model.parameters(), 
                                          lr=self.config["TRAINING"]["Learning_rate"],
                                          weight_decay=self.config["TRAINING"]["Weight_decay"])

        self.trainer = TrainerCurriculumCNF(
            model = self.model,
            optimizer = self.optimizer,
            scheduler = self.scheduler,
            loss_function = self.loss_function,
            epoch_loader = self.epoch_loader,
            evaluation_functions = {},
            n_epochs = self.config["BASCIC"]["N_epochs"],
            early_stopping_patience = self.config["TRAINING"]["Early_stopping_patience"],
            schedule_step_on = "batch",
            save_path = self.config["BASCIC"]["Save_path"],
            coupling = None,
            use_same_timestep_per_batch = False,
            use_train_mode_during_validation = False,
            max_gradient_norm = self.config["TRAINING"]["max_grad_norm"],

        )

        initial_val_loss = self.trainer.validate()

        print(f"Initial validation loss: {initial_val_loss}")

        r = self.trainer.train()

        self.trainer.load_best_model()
        self.model = self.trainer.model
        self.model.eval()

        test_res = self.trainer.test()

        print(f"Test results: {test_res}")

    def setup_full_model(self):
        """
        Setup the full model.
        """

        self.full_model = ModelToPosteriorCNF(
            **self.config["FULL_MODEL"]
        )

    def setup_evaluation(self):
        """
        Setup the evaluation.
        """

        benchmark_params_ppgrogram = self.data_generator.curriculum.get_params(-1)
        print(f"params for pprogram: {benchmark_params_ppgrogram}")

        self.pprogram1 = self.config["DATA_GENERATION"]["Pprogram"]

        assert self.pprogram1 is not None, "pprogram1 is None!"

        self.pprogram1_y = return_only_y(self.pprogram1)
        
        print_code(self.pprogram1)

        if self.config["Evaluation"]["do_full_evaluation"] is True: 
            self.comparison_models = make_default_list_comparison(
                pprogram_y=self.pprogram1_y,
                n_samples=self.config["Evaluation"]["N_samples_per_model"]
            )
        else:
            self.comparison_models = make_reduced_list_comparison(
                pprogram_y=self.pprogram1_y,
                n_samples=self.config["Evaluation"]["N_samples_per_model"]
            )

        


    def evaluate_synthetic(self):
        """
        Evaluate the synthetic data.
        """
        use_intercept = self.config["DATA_GENERATION"]["Use_intercept"]

        self.evaluator = Evaluate(
            posterior_model=self.full_model,
            evaluation_loader=self.trainer.testset,
            comparison_models=self.comparison_models,
            n_evaluation_cases = self.config["Evaluation"]["N_synthetic_cases"],
            save_path = self.config["BASCIC"]["Save_path"],
            results_dict_to_latent_variable_comparison_models = just_return_results if not use_intercept else results_dict_to_latent_variable_beta0_and_beta,
        )

        self.eval_res_synthetic = self.evaluator.evaluate()

    def evaluate_real_world(self):
        """
        Evaluate the real world data.
        """

        target_mean = self.check_model_res[1]["y"]['mean_mean']
        target_var = self.check_model_res[1]["y"]['variance_mean']

        self.getdata = GetDataOpenML(
            preprocessor = Preprocessor(
                N_datapoints = self.config["BASIC"]["N"],
                P_features = self.config["BASIC"]["P"],
                target_mean = target_mean,
                target_var = target_var
            ),
            save_path = self.config["EVALUATION"]["save_path_data_real_world_eval"],
            benchmark_id = self.config["EVALUATION"]["real_world_benchmark_id"]
        )
        self.datasets = self.getdata.get_data()

        use_intercept = self.config["DATA_GENERATION"]["Use_intercept"]

        if self.config["n_evaluation_cases_real_world"] == "All":
            n_evaluation_cases = len(self.datasets)

        self.eval_rw = EvaluateRealWorld(
            posterior_model = self.full_model,
            evaluation_datasets = self.datasets,
            comparison_models = self.comparison_models,
            results_dict_to_latent_variable_comparison_models = just_return_results if not use_intercept else results_dict_to_latent_variable_beta0_and_beta,
            n_evaluation_cases = n_evaluation_cases,
            save_path = self.config["BASCIC"]["Save_path"],
            overwrite_results = True
        )

        self.eval_res_real_world = self.eval_rw.evaluate()

        self.eval_rw.plot_results()


        
        