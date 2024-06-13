import torch 
import ot 

class MiniBatchOTCoupling:
    """
    Class to couple z_0 and z_t with  optimal transport
    """

    def __init__(
            self,
            solver:str = "sinkhorn",
            solver_params: dict = {
                "reg": 1e-10,
                "numItermax": 1000
                }
    ):
        """
        Args:
            solver: str: the solver to use for the optimal transport problem
            solver_params: dict: the parameters for the solver
        """
        assert solver in ["sinkhorn", "emd"], f"solver {solver} not implemented. Choose from ['sinkhorn', 'emd']"

        self.solver = solver
        self.solver_params = solver_params

    
    def compute_similarity_mat(self, z_0: torch.tensor, z_t: torch.tensor) -> torch.tensor:
        """
        Compute the similarity matrix between z_0 and z_t using the squared L2 norm
        Args:
            z_0: torch.tensor: the tensor from the base distribution of shape (batch_size, n_features)
            z_t: torch.tensor: the tensor from the data distribution of shape (batch_size, n_features)

        Returns:
        """
        assert z_0.shape == z_t.shape, f"z_0 shape {z_0.shape} does not match z_t shape {z_t.shape}"

        sim_mat = torch.cdist(z_0, z_t, p=2)**2 # (batch_size, batch_size)

        return sim_mat
    
    def compute_coupling(self, z_0: torch.tensor, z_t: torch.tensor) -> torch.tensor:
        """
        Compute the coupling between z_0 and z_t
        Args:
            z_0: torch.tensor: the tensor from the base distribution of shape (batch_size, n_features)
            z_t: torch.tensor: the tensor from the data distribution of shape (batch_size, n_features)

        Returns:
            coupling: torch.tensor: the coupling between z_0 and z_t of shape (batch_size, batch_size)
        """
        sim_mat = self.compute_similarity_mat(z_0, z_t)

        cost_mat = torch.max(sim_mat) - sim_mat

        a = torch.ones(z_0.shape[0])/z_0.shape[0] #uniform weights on samples from z_0 and z_t
        a = a.to(z_0.device)
        if self.solver == "sinkhorn":
            coupling = ot.sinkhorn(a, a, cost_mat, **self.solver_params)
        elif self.solver == "emd":
            coupling = ot.emd(a, a, cost_mat, **self.solver_params)
        
        return coupling
    
    def get_assignment_from_coupling(self, coupling: torch.tensor) -> torch.tensor:
        """
        Get the assignment from the coupling
        Args:
            coupling: torch.tensor: the coupling between z_0 and z_t of shape (batch_size, batch_size)

        Returns:
            assignment: torch.tensor: the assignment from the coupling of shape (batch_size,)
        """
        assignment = coupling.argmax(dim=1)

        # check for duplicates
        assert len(assignment) == len(set(assignment)), "Duplicates in the assignment"

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

        assignment = self.get_assignment_from_coupling(coupling)

        z_t_coupled = z_t[assignment]

        return z_t_coupled