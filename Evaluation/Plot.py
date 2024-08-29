import matplotlib.pyplot as plt
import seaborn as sns
import torch 

from PFNExperiments.Evaluation.CompareModelToGT import results_dict_to_data_x_y, results_dict_to_latent_variable_beta, flatten_dict_list



class Plot:
    """
    A class to make several plot regarding the results of the experiments
    """

    def __init__(
        self,
        results_to_dict_data_x_y: callable = results_dict_to_data_x_y,
        results_to_dict_latent_variable: callable = results_dict_to_latent_variable_beta,
        save_path: str = None,
        fontsize: int = 12,
    ):
        """
        Args:
            results_to_dict_data_x_y: callable: a function that takes the results dictionary and returns the data
            results_to_dict_latent_variable: callable: a function that takes the results dictionary and returns the latent variable
            save_path: str: the path to save the plots
            fontsize: int: the fontsize of the plots
        """

        self.results_to_dict_data_x_y = results_to_dict_data_x_y
        self.results_to_dict_latent_variable = results_to_dict_latent_variable
        self.save_path = save_path
        self.fontsize = fontsize


    def density_plot_marginals(
            self,
            model_samples: dict[list[dict]],
            gt_samples: list[dict],
            plot_gt: bool = True,
            max_number_plots: int = 5,
    ) -> None:
        """
        A function to plot the density of the samples by the models
        Args:
            model_samples: dict[list[dict]]: the samples by the models
            gt_samples: list[dict]: the ground truth samples
            plot_gt: bool: whether to plot the ground truth samples
            max_number_plots: int: the maximum number of plots to show
        """

        n_cases = len(model_samples[list(model_samples.keys())[0]])

        if gt_samples is not None:
            assert len(gt_samples) == n_cases, "The number of cases in the ground truth samples is different from the number of cases in the model samples"

       
        max_number_plots = min(max_number_plots, n_cases)
        # draw without replacement
        random_indices = torch.multinomial(torch.ones(n_cases), max_number_plots, replacement = False)

        # define a model to color mapping with the default color palette
        model_color = {model: sns.color_palette()[i] for i, model in enumerate(model_samples.keys())}
        model_color["GT"] = "black"


        for i in random_indices:
            samples_per_model = {}

            for model, samples in model_samples.items():
                samples_per_model[model] = self.results_to_dict_latent_variable(samples[i])

            if gt_samples is not None:
                gt_parameter = self.results_to_dict_latent_variable(gt_samples[i]).squeeze()
                gt_parameter = gt_parameter.cpu().detach().numpy()

            n_dims = samples_per_model[model].shape[1]

            fig, ax = plt.subplots(ncols = n_dims, figsize = (5*n_dims, 5))

            for j in range(n_dims):
                for model, samples in samples_per_model.items():
                    try:
                        samples = samples.cpu().detach().numpy()
                    except:
                        pass
                    sns.kdeplot(samples[:, j], ax = ax[j], color = model_color[model])
                
                if gt_samples is not None and plot_gt:
                    ax[j].axvline(gt_parameter[j], color = model_color["GT"])
                ax[j].set_title(f"Dimension {j}", fontsize = self.fontsize)

            # set font size
            for axis in ax:
                #axis.tick_params(labelsize = self.fontsize)
                #axis.set_xlabel("Density", fontsize = self.fontsize)
                #axis.set_ylabel("Value", fontsize = self.fontsize)
                pass



            # set legend once for the figure because it is the same for all subplots. Set it below the subplots by using figlegend

            handles = [plt.Line2D([0], [0], color = model_color[model], label = model) for model in model_samples.keys()]
            if gt_samples is not None and plot_gt:
                handles.append(plt.Line2D([0], [0], color = "black", label = "GT"))
            fig.legend(handles=handles, loc='center left', fontsize=self.fontsize, bbox_to_anchor=(-1.0, 0.5))
            # set title 
            fig.suptitle(f"Dataset Id: {i}", fontsize = self.fontsize)
          
            # use tight layout

            fig.tight_layout(rect = [0, 0.03, 1, 0.95])
        

            if self.save_path is not None:
                plt.savefig(f"{self.save_path}/density_plot_example_{i}.png", dpi = 300)


            plt.show()

        