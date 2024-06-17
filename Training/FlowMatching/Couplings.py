import torch 
import ot 

class MiniBatchOTCoupling:
    """
    Class to couple z_0 and z_t with  optimal transport
    """

    def __init__(
            self,
            solver:str = "emd",
            solver_params: dict = {
                #"reg": 1e-10,
                #"numItermax": 1000
                },
            sample_with_replacement: bool = True
    ):
        """
        Args:
            solver: str: the solver to use for the optimal transport problem
            solver_params: dict: the parameters for the solver
        """
        assert solver in ["sinkhorn", "emd"], f"solver {solver} not implemented. Choose from ['sinkhorn', 'emd']"

        self.solver = solver
        self.solver_params = solver_params
        self.replace = sample_with_replacement

        
    def compute_coupling(self, z_0: torch.tensor, z_t: torch.tensor) -> torch.tensor:
        """
        Compute the coupling between z_0 and z_t
        Args:
            z_0: torch.tensor: the tensor from the base distribution of shape (batch_size, n_features)
            z_t: torch.tensor: the tensor from the data distribution of shape (batch_size, n_features)

        Returns:
            coupling: torch.tensor: the coupling between z_0 and z_t of shape (batch_size, batch_size)
        """
        dist_mat = ot.dist(z_0, z_t, metric="euclidean") # (batch_size, batch_size)

        a = torch.ones(z_0.shape[0])/z_0.shape[0] #uniform weights on samples from z_0 and z_t
        a = a.to(z_0.device)
        if self.solver == "sinkhorn":
            coupling = ot.bregman.sinkhorn(a, a, dist_mat, **self.solver_params)
        elif self.solver == "emd":
            coupling = ot.emd(a, a, dist_mat, **self.solver_params)
        
        return coupling
    
    def get_assignment_from_coupling_argmax(self, coupling: torch.tensor) -> torch.tensor:
        """
        Get the assignment from the coupling by just assigning each element to the best match
        Args:
            coupling: torch.tensor: the coupling between z_0 and z_t of shape (batch_size, batch_size)

        Returns:
            assignment: torch.tensor: the assignment from the coupling of shape (batch_size,)
        """
        assignment = coupling.argmax(dim=1)

        # check for duplicates
        assert len(assignment) == len(set(assignment)), "Duplicates in the assignment"

        return assignment
    
    def get_assignment_from_coupling_sampling(self, coupling: torch.tensor) -> torch.tensor:
        """
        Get the assignment from the coupling
        Args:
            coupling: torch.tensor: the coupling between z_0 and z_t of shape (batch_size, batch_size)

        Returns:
            assignment: torch.tensor: the assignment from the coupling of shape (batch_size,)
        """
        assignment = torch.multinomial(coupling, 1, replacement=self.replace).squeeze()

        return assignment
    
    def couple(self, z_0: torch.tensor, z_t: torch.tensor) -> torch.tensor:
        """
        Couple z_0 and z_t
        Args:
            z_0: torch.tensor: the tensor from the base distribution of shape (batch_size, n_features)
            z_t: torch.tensor: the tensor from the data distribution of shape (batch_size, n_features)

        Returns:
            z_t_c: torch.tensor: the coupled tensor from the base distribution of shape (batch_size, n_features)
        """
        coupling = self.compute_coupling(z_0, z_t)

        assignment = self.get_assignment_from_coupling_sampling(coupling)

        z_t_coupled = z_t[assignment]

        return z_t_coupled