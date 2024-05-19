import matplotlib.pyplot as plt

class Curriculum():
    """
    A curriculum that determines the way the samples are generated.

    Attributes:
        max_iter: int: the maximum number of iterations for training the model (in mini-batches)
        generation_params: dict: a dictionary containing the parameters of the curriculum. The keys are the names of the parameters and the values are the schedule functions.
    """

    def __init__(self,
                 max_iter:int = None,
                 n_epochs: int = None,
                 n_iterations_per_epoch: int = None,
                 ):
        
        """
        Args: 
            max_iter: int: the maximum number of iterations for training the model (in mini-batches), default None
            n_epochs: int: the number of epochs, default None
            n_iterations_per_epoch: int: the number of iterations per epoch, default None
        """
        if max_iter is None and n_epochs is None and n_iterations_per_epoch is None:
            raise ValueError("max_iter, n_epochs, and n_iterations_per_epoch cannot be all None")
        if max_iter is not None:
            self.max_iter = max_iter
        else:
            self.max_iter = n_epochs * n_iterations_per_epoch


        self.max_iter = max_iter
        self.generation_params = {}

    def __repr__(self) -> str:
        return f"Curriculum(max_iter={self.max_iter}, generation_params={self.generation_params})"


    def linear_scheduler(self, start_value:float, end_value:float) -> callable:
        """
        A linear scheduler that linearly interpolates between start_value and end_value.
        Args:
            start_value: float: the start value
            end_value: float: the end value
        Returns:
            callable: the schedule function
        """
        def schedule(iter:int) -> float:
            return start_value + (end_value - start_value) * iter / self.max_iter

        return schedule
    
    def constant_scheduler(self, value:float) -> callable:
        """
        A constant scheduler that returns the same value.
        Args:
            value: float: the value
        Returns:
            callable: the schedule function
        """
        def schedule(iter:int) -> float:
            return value

        return schedule
    
    def stepwise_scheduler(self, start_value:float, end_value:float, n_steps:int = 10) -> callable:
        """
        A stepwise scheduler that changes the value every n_steps iterations.
        Args:
            start_value: float: the start value
            end_value: float: the end value
            n_steps: int: the number of steps
        Returns:
            callable: the schedule function
        """
        def schedule(iter:int) -> float:
            return start_value + (end_value - start_value) * (iter // n_steps) / self.max_iter

        return schedule
    
    def constant_than_linear_scheduler(self, start_value:float, end_value:float, fraction_constant:float = 0.3) -> callable:
        """
        A scheduler that is constant for a fraction of the iterations and then linearly interpolates between start_value and end_value.
        Args:
            start_value: float: the start value
            end_value: float: the end value
            fraction_constant: float: the fraction of iterations that the value is constant
        Returns:
            callable: the schedule function
        """
        def schedule(iter:int) -> float:
            if iter < self.max_iter * fraction_constant:
                return start_value
            else:
                return start_value + (end_value - start_value) * (iter - self.max_iter * fraction_constant) / (self.max_iter * (1 - fraction_constant))
        return schedule
    
    def constant_than_linear_than_constant_scheduler(self, 
                                                     start_value:float, 
                                                     end_value:float, 
                                                     fraction_constant_beginning:float = 0.2, 
                                                     fraction_constant_end:float = 0.2) -> callable:
        """
        A scheduler that is constant for a fraction of the iterations, then linearly interpolates between start_value and end_value, and then is constant again.
        Args:
            start_value: float: the start value
            end_value: float: the end value
            fraction_constant_beginning: float: the fraction of iterations that the value is constant at the beginning
            fraction_constant_end: float: the fraction of iterations that the value is constant at the end
        Returns:
            callable: the schedule function
        """
        def schedule(iter:int) -> float:
            if iter < self.max_iter * fraction_constant_beginning:
                return start_value
            elif iter < self.max_iter * (1 - fraction_constant_end):
                return start_value + (end_value - start_value) * (iter - self.max_iter * fraction_constant_beginning) / (self.max_iter * (1 - fraction_constant_beginning - fraction_constant_end))
            else:
                return end_value
            
        return schedule
    
    
    def exponential_scheduler(self, start_value:float, end_value:float) -> callable:
        """
        An exponential scheduler that exponentially interpolates between start_value and end_value.
        Args:
            start_value: float: the start value
            end_value: float: the end value
        Returns:
            callable: the schedule function
        """
        def schedule(iter:int) -> float:
            return start_value + (end_value - start_value) * (1 - 2 ** (-iter / self.max_iter))

        return schedule


    def plot_schedule(self, schedule:callable, steps_to_plot = 1000) -> None:
        """
        Plot the schedule
        Args:
            schedule: callable: the schedule function
            steps_to_plot: int: the number of steps to plot
        """

        plot_range = range(0, self.max_iter, self.max_iter // steps_to_plot)


        values = [schedule(i) for i in plot_range]
        plt.plot(values)
        plt.show()


    def add_param(self, name:str, schedule:callable) -> None:
        """
        Add a parameter to the curriculum
        Args:
            name: str: the name of the parameter
            schedule: callable: the schedule function
        """
        self.generation_params[name] = schedule

    def add_param_list(self, lis: list) -> None:
        """
        Add a list of parameters to the curriculum
        Args:
            list(tuple): a list of tuples containing the name of the parameter and the schedule function
        """
        for name, schedule in lis:
            self.add_param(name, schedule)

    def plot_all_schedules(self, steps_to_plot = 1000) -> None:
        """
        Plot all schedules
        Args: 
            steps_to_plot: int: the number of steps to plot
        """
        fig, axs = plt.subplots(len(self.generation_params), 1)
        fig.set_figheight(5 * len(self.generation_params))
        for i, (name, schedule) in enumerate(self.generation_params.items()):
            values = [schedule(i) for i in range(0, self.max_iter, self.max_iter // steps_to_plot)]
            axs[i].plot(values)
            axs[i].set_title(name)
            axs[i].x_values = range(0, self.max_iter, self.max_iter // steps_to_plot)
            axs[i].set_xlabel("Iteration")


        plt.show()

    def get_params(self, iter:int) -> dict:
        """
        Get the parameters at a certain iteration
        Args:
            iter: int: the iteration
        Returns:
            dict: the parameters
        """
        if iter == -1:
            iter = self.max_iter - 1

        return {name: schedule(iter) for name, schedule in self.generation_params.items()}
    
    def __call__(self, iter:int) -> dict:
        return self.get_params(iter)
    
    def __getitem__(self, iter:int) -> dict:
        return self.get_params(iter)   
    
    def __len__(self) -> int:
        return self.max_iter