import torch 


def batch_to_model_lm(batch:dict) -> torch.tensor:
    x = batch['x']
    y = batch['y']
    X_y = torch.cat([x, y.unsqueeze(-1)], dim = -1)
    return X_y


class Trainer():
    """
    A custom class for training neural networks
    """

    def __init__(self,
                 model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 loss_function: torch.nn.modules.loss._Loss,
                 trainset: torch.utils.data.Dataset,
                 valset: torch.utils.data.Dataset,
                 batch_to_model_function: callable = batch_to_model_lm,
                 target_key_in_batch: str = "beta",
                 evaluation_functions: dict[str, callable] = {},
                 device: torch.device = torch.device("cude" if torch.cuda.is_available() else "cpu"),
                 n_epochs: int = 100,
                 save_path: str = "../models/model.pth",
                 early_stopping_patience: int = 10,
    ):
        """
        A custom class for training neural networks
        Args:
            model: torch.nn.Module: the model
            optimizer: torch.optim.Optimizer: the optimizer
            loss_function: torch.nn.modules.loss._Loss: the loss function
            trainset: torch.utils.data.Dataset: the training set
            valset: torch.utils.data.Dataset: the validation set
            batch_to_model_function: callable: a function that maps a batch to the model input
            target_key_in_batch: str: the key of the target in the batch
            evaluation_functions: dict[str, callable]: the evaluation functions in a dictionary where the key is the name of the evaluation function and the value is the evaluation function
            device: torch.device: the device
            n_epochs: int: the number of epochs
            save_path: str: the path to save the model
            early_stopping_patience: int: the patience for early stopping
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.trainset = trainset
        self.valset = valset
        self.batch_to_model_function = batch_to_model_function
        self.evaluation_functions = evaluation_functions
        self.target_key_in_batch = target_key_in_batch
        self.device = device
        self.n_epochs = n_epochs
        self.save_path = save_path
        self.early_stopping_patience = early_stopping_patience

        self.evaluation_functions["loss"] = self.loss_function 


    def validate(self):
        """
        Validate the model
        """
        self.model.eval()

        predictions = []
        targets = []

        with torch.no_grad():
            for batch in self.valset:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                x = self.batch_to_model_function(batch)
                target = batch[self.target_key_in_batch]
                pred = self.model(x)

                predictions.append(pred.detach().cpu())
                targets.append(target.detach().cpu())

            predictions = torch.cat(predictions, dim=0)
            targets = torch.cat(targets, dim=0)

            results = {}
            for name, fun in self.evaluation_functions.items():
                results[name] = fun(predictions, targets)

        
                