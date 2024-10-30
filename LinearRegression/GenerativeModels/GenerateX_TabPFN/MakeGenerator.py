import torch 
from PFNExperiments.LinearRegression.GenerativeModels.GenerateX import make_simulate_X_by_loading



class MakeGenerator():

    def __init__(self,
                 paths: list = [
                     "/content/drive/MyDrive/PFN_Experiments/DataX/TabPFN_Prior/X_tabpfn_n50_p5_1_000_000_v3_scaled.pt",
                     "/content/drive/MyDrive/PFN_Experiments/DataX/TabPFN_Prior/X_tabpfn_n50_p5_2_000_000_v4_scaled.pt",
                     "/content/drive/MyDrive/PFN_Experiments/DataX/TabPFN_Prior/X_tabpfn_n50_p5_2_000_000_v5_scaled.pt",
                     "/content/drive/MyDrive/PFN_Experiments/DataX/TabPFN_Prior/X_tabpfn_n50_p5_2_000_000_v6_scaled.pt",
                    "content/drive/MyDrive/PFN_Experiments/DataX/TabPFN_Prior/X_tabpfn_n50_p5_2_000_000_v7_scaled.pt",
                 ],
                 set_nan_to_zero: bool = True,):
        """
        Make the X-generator with TabPFN data that is loaded from a file.

        Args:
        paths: list: the paths to the data files
        set_nan_to_zero: bool: set NaN values to zero
        """
        self.paths = paths
        self.set_nan_to_zero = set_nan_to_zero

    def load_data(self):
        """
        Only load the data to obtain X_tabpfn.
        """

        with open(self.paths[0], "rb") as f:
            X_tabpfn = torch.load(f)

        final_res_ten = torch.zeros(X_tabpfn.shape)

        for data_path in self.paths:
            with open(data_path, "rb") as f:
                X_tabpfn_new = torch.load(f)

            final_res_ten = torch.cat((final_res_ten, X_tabpfn_new), dim=0)


        X_tabpfn = final_res_ten

        
        print(f"got {X_tabpfn.isnan().sum()} NaNs")
        print(f"got shape {X_tabpfn.shape}")

        X_tabpfn[torch.isnan(X_tabpfn)] = 0.0

        return X_tabpfn

    def make_generator(self) -> callable:
        """
        Make the generator function for the X data.
        Returns
        callable: the generator function
        """

        with open(self.paths[0], "rb") as f:
            X_tabpfn = torch.load(f)

        final_res_ten = torch.zeros(X_tabpfn.shape)

        for data_path in self.paths:
            with open(data_path, "rb") as f:
                X_tabpfn_new = torch.load(f)

            final_res_ten = torch.cat((final_res_ten, X_tabpfn_new), dim=0)


        X_tabpfn = final_res_ten

        
        print(f"got {X_tabpfn.isnan().sum()} NaNs")
        print(f"got shape {X_tabpfn.shape}")

        X_tabpfn[torch.isnan(X_tabpfn)] = 0.0

        generate_X_tabpfn = make_simulate_X_by_loading(X_tabpfn)

        return generate_X_tabpfn