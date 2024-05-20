import torch 
from tqdm import tqdm
import time
import os
import matplotlib.pyplot as plt
from typing import Tuple

try:
    from Trainer import batch_to_model_lm, visualize_training_results

except:
    from PFNExperiments.Training.Trainer import batch_to_model_lm, visualize_training_results

class TrainerCurriculum():
    """
    A custom class for training neural networks
    """

    def __init__(self,
                 model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 loss_function: torch.nn.modules.loss._Loss,
                 epoch_loader: callable,
                 valset: torch.utils.data.Dataset = None,
                 testset: torch.utils.data.Dataset = None,
                 scheduler: torch.optim.lr_scheduler = None,
                 schedule_step_on: str = "epoch",
                 batch_to_model_function: callable = batch_to_model_lm,
                 target_key_in_batch: str = "beta",
                 evaluation_functions: dict[str, callable] = {},
                 device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 n_epochs: int = 100,
                 save_path: str = "../models/model.pth",
                 early_stopping_patience: int = 10,
                 verbose:int = 100
    ):
        """
        A custom class for training neural networks
        Args:
            model: torch.nn.Module: the model
            optimizer: torch.optim.Optimizer: the optimizer
            loss_function: torch.nn.modules.loss._Loss: the loss function
            epoch_loader: callable: a function that returns the training, testing and validation set for each epoch according to the curriculum
            valset: torch.utils.data.Dataset: the validation set
            testset: torch.utils.data.Dataset: the test set
            scheduler: torch.optim.lr_scheduler: the scheduler
            schedule_step_on: str: the step on which to schedule the learning rate. Can be "epoch" or "batch"
            batch_to_model_function: callable: a function that maps a batch to the model input
            target_key_in_batch: str: the key of the target in the batch
            evaluation_functions: dict[str, callable]: the evaluation functions in a dictionary where the key is the name of the evaluation function and the value is the evaluation function
            device: torch.device: the device
            n_epochs: int: the number of epochs
            save_path: str: the path to save the model
            early_stopping_patience: int: the patience for early stopping
            verbose: int: how much to print
        """

        assert schedule_step_on in ["epoch", "batch"], "schedule_step_on must be either 'epoch' or 'batch'"

        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.epoch_loader = epoch_loader
        self.valset = valset
        self.testset = testset
        self.scheduler = scheduler
        self.schedule_step_on = schedule_step_on
        self.batch_to_model_function = batch_to_model_function
        self.evaluation_functions = evaluation_functions
        self.target_key_in_batch = target_key_in_batch
        self.device = device
        self.n_epochs = n_epochs
        self.save_path = save_path
        self.early_stopping_patience = early_stopping_patience
        self.verbose = verbose

        if self.valset is None:
            self.valset = self.epoch_loader(0)[1]
        
        if self.testset is None:
            self.testset = self.epoch_loader(self.n_epochs)[2]

        self.model = self.model.to(self.device)

        if not os.path.exists(os.path.dirname(self.save_path)):
            os.makedirs(os.path.dirname(self.save_path))

    def validate_loader(self, loader: torch.utils.data.DataLoader) -> dict[str, float]:
        """
        Validate the model
        Args:
            loader: torch.nn.utils.data.DataLoader: the loader
        Returns
            dict[str, float]: the validation results where the key is the name of the evaluation function and the value is the result of the evaluation function
        """
        self.model.eval()

        predictions = []
        targets = []

        with torch.no_grad():
            for batch in tqdm(loader):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                x = self.batch_to_model_function(batch)
                target = batch[self.target_key_in_batch]
                pred = self.model(x)

                predictions.append([elem.detach().cpu() for elem in pred])
                targets.append(target.detach().cpu())

        
        results = {}
        for name, fun in self.evaluation_functions.items():
            results[name] = fun(targets, predictions)

        
        return results

    def validate(self) -> dict[str, float]:
        """
        Validate the model
        Returns
            dict[str, float]: the validation results where the key is the name of the evaluation function and the value is the result of the evaluation function
        """
        self.model.eval()

        predictions = []
        targets = []

        with torch.no_grad():
            for batch in tqdm(self.valset):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                x = self.batch_to_model_function(batch)
                target = batch[self.target_key_in_batch]
                pred = self.model(x)

                predictions.append([elem.detach().cpu() for elem in pred])
                targets.append(target.detach().cpu())

        
            results = {}
            for name, fun in self.evaluation_functions.items():
                results[name] = fun(targets, predictions)

        
        return results
    
    def test(self) -> dict[str, float]:
        """
        Test the model
        Returns
            dict[str, float]: the test results where the key is the name of the evaluation function and the value is the result of the evaluation function
        """
        self.model.eval()

        predictions = []
        targets = []

        with torch.no_grad():
            for batch in tqdm(self.testset):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                x = self.batch_to_model_function(batch)
                target = batch[self.target_key_in_batch]
                pred = self.model(x)

                predictions.append([elem.detach().cpu() for elem in pred])
                targets.append(target.detach().cpu())

        
            results = {}
            for name, fun in self.evaluation_functions.items():
                results[name] = fun(targets, predictions)

        
        return results
    
                
    def train(self) -> tuple[list[dict[str, float]], list[dict[str, float]], list[float]]:
        """
        Train the model
        Returns:
            tuple[list[dict[str, float]], list[dict[str, float]], list[float]]: a tuple containing the training results, the validation results and the time for each epoch
        """
        best_val_loss = float("inf")
        patience = 0

        self.model = self.model.to(self.device)

        all_results_training = []
        all_results_validation = []
        all_results_validation_curriculum = []  # the validation results for the current curriculum
        all_results_time = []
        
        start_time_epoch = time.time()
        overall_iter = 0

        batch_size = 0
        for epoch in range(self.n_epochs):
            self.model.train()
            predictions = []
            targets = []

            trainset_epoch = self.epoch_loader(epoch)[0]
            valset_epoch = self.epoch_loader(epoch)[1]

            
            if self.verbose > 1: 
                current_params = self.epoch_loader.GenerateDataCurriculum.curriculum.get_params(iter = overall_iter * batch_size)
                print(f"Curriculum parameters: {current_params} at iteration { overall_iter * batch_size}")

            for batch in tqdm(trainset_epoch):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                x = self.batch_to_model_function(batch)

                batch_size = x.shape[0]

                target = batch[self.target_key_in_batch]
                pred = self.model(x)

                loss = self.loss_function(pred, target)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if self.scheduler is not None and self.schedule_step_on == "batch":
                    try: 
                        self.scheduler.step()
                    except:
                        self.scheduler.step(loss)

                predictions.append([elem.detach().cpu() for elem in pred])
                targets.append(target.detach().cpu())

                overall_iter += 1

        
            validation_results = self.validate()

            validation_results_current_curriculum = self.validate_loader(valset_epoch)

            if self.scheduler is not None and self.schedule_step_on == "epoch":
                try: 
                    self.scheduler.step()
                except:
                    self.scheduler.step(validation_results["loss"])
            training_results = {}
            for name, fun in self.evaluation_functions.items():
                training_results[name] = fun(targets, predictions)

            end_time_epoch = time.time()
            time_epoch = end_time_epoch - start_time_epoch

            all_results_training.append(training_results)
            all_results_validation.append(validation_results)
            all_results_validation_curriculum.append(validation_results_current_curriculum)
            all_results_time.append(time_epoch)

            print(f"Epoch {epoch}:")
            print(f"Training: {training_results}")
            print(f"Validation: {validation_results}")
            print(f"Validation curriculum: {validation_results_current_curriculum}")
            print(f"Time: {time_epoch}")
            if self.scheduler is not None:
                try:
                    print(f"Learning rate: {self.scheduler.get_last_lr()}")
                except:
                    try: 
                        print(f"Learning rate: {self.scheduler._last_lr}")
                    except:
                        print("Could not get learning rate")
                
            print("\n")
            print(100*"-")

            val_loss = validation_results_current_curriculum["loss"]
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print("Saving model")
                torch.save(self.model.state_dict(), self.save_path)
                patience = 0
            else:
                patience += 1

            if patience > self.early_stopping_patience:
                print("Early stopping")
                break

            start_time_epoch = time.time()

        
        if self.testset is not None:
            test_results = self.test()
            print(f"Test: {test_results}")
            return all_results_training, all_results_validation, all_results_time, test_results
        

        return all_results_training, all_results_validation, all_results_time
    

    def load_best_model(self) -> None:
        """
        Load the best model
        """
        self.model.load_state_dict(torch.load(self.save_path))

def visualize_training_results(results: Tuple[list[dict[str, float]], list[dict[str, float]]], loglog = True):
    """
    visualize results as returned by train()
    Args: 
        results: Tuple[list[dict[str, float]], list[dict[str, float]]]: the results
        loglog: bool: whether to use loglog scale
    """

    n_subplots = len(results[0][0])

    if n_subplots == 1:
        fig, axs = plt.subplots(1, 1, figsize=(10, 10))
        
        training = [result["loss"] for result in results[0]]
        validation = [result["loss"] for result in results[1]]

        if loglog:
            axs.plot(training, label="Training")
            axs.plot(validation, label="Validation")
            axs.set_title("Loss")
            axs.set_yscale("log")
            axs.set_xscale("log")
            axs.legend()
            return
        else:
            axs.plot(training, label="Training")
            axs.plot(validation, label="Validation")
            axs.set_title("Loss")
            axs.legend()
        return 


    fig, axs = plt.subplots(n_subplots, 1, figsize=(10, 10*n_subplots))

    for i, (name, fun) in enumerate(results[0][0].items()):
        training = [result[name] for result in results[0]]
        validation = [result[name] for result in results[1]]

        if loglog:
            axs[i].plot(training, label="Training")
            axs[i].plot(validation, label="Validation")
            axs[i].set_title(name)
            axs[i].set_yscale("log")
            axs[i].set_xscale("log")
            axs[i].legend()

        else:
            axs[i].plot(training, label="Training")
            axs[i].plot(validation, label="Validation")
            axs[i].set_title(name)
            axs[i].legend()