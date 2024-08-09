import openml
from tqdm import tqdm
import pickle
from PFNExperiments.Evaluation.RealWorldEvaluation.PreprocessDataset import Preprocessor

class GetDataOpenML:
    """
    A class to get the data from OpenML
    """
    def __init__(
            self, 
            benchmark_id: int = 336,  #   Tabular benchmark numerical regression,
            preprocessor = Preprocessor(
                N_datapoints = 100,
                P_features = 10
                ),
            save_path: str = None
    ):
        """
        Args:
            benchmark_id: int: the OpenML benchmark id
            preprocessor: callable: a preprocessor
            save_path: str: the path to save the data
        """
        self.benchmark_id = benchmark_id
        self.preprocessor = preprocessor
        self.save_path = save_path


    def fetch_data_openml(self):
        """
        download the data from OpenML

        Returns:
            list[dict]: the data
        """

        print("Downloading the data from OpenML")

        benchmark_suite = openml.study.get_suite(self.benchmark_id)  #   Tabular benchmark categorical classification

        res_lis = []
        for task_id in tqdm(benchmark_suite.tasks):  # iterate over all tasks
            task = openml.tasks.get_task(task_id)  # download the OpenML task
            features, targets = task.get_X_and_y()  # get the data

            res_lis.append({
                "id": task_id,
                "name": str(task),
                "features": features,
                "targets": targets
            })

        dataset_lis = [
            {   
                "id": res["id"],
                "x": res["features"],
                "y": res["targets"]
            }
            for res in res_lis
        ]

        if self.save_path is not None:
            with open(self.save_path, "wb") as f:
                pickle.dump(dataset_lis, f)

        return dataset_lis
    
    def load_data_from_disk(self):
        """
        Load the data from disk
        Returns:
            list[dict]: the data
        """
        with open(self.save_path, "rb") as f:
            dataset_lis = pickle.load(f)
        
        return dataset_lis
        


    def get_data(self):
        """
        Get the data from OpenML
        Returns:
            list[dict]: the data
        """
        try:
            dataset_lis =  self.load_data_from_disk()
        except Exception as e:
            print(f"The data is not loaded from disk. Error message: {e}. Fetching the data from OpenML")
            dataset_lis = self.fetch_data_openml()
        

        new_dataset_lis = []
        for dataset in dataset_lis:

            try:
                id = dataset["id"]
                dataset = self.preprocessor.preprocess(dataset)
                dataset["id"] = id
                
            except Exception as e:
                print(f"An error {e} occured while preprocessing the dataset with id {dataset['id']}. Skipping the dataset")
                continue

            new_dataset_lis.append(dataset)

        return new_dataset_lis
    
