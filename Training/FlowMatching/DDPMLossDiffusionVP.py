import torch 

class DDPMLossDiffusionVP(torch.nn.Module):
    """
    Implementation of the DDPMLoss with variance preserving paths using score matching
    """

    def __init__(
            self,
            epsilon_for_t: float = 1e-5,
            beta_min: float = 0.1,
            beta_max: float = 20,
    ):
        """
        
        epsilon_for_t: float: the epsilon for the time step, i.e. time has to be in [0, 1 - epsilon_for_t]
        beta_min: float: the minimum value of the beta
        beta_max: float: the maximum value of the beta
        """

        super(DDPMLossDiffusionVP, self).__init__()
        self.epsilon_for_t = epsilon_for_t
        self.beta_min = beta_min
        self.beta_max = beta_max

    def T_t(
            self,
            t: float,
    ):
        """
        Compute the t_T function
        Args:
            t: float: the time step has shape (batch_size, 1)
        """
        return t * self.beta_min + 0.5*(t**2)*(self.beta_max - self.beta_min)
    
    def T_t_prime(
            self,
            t: float,
    ):
        """
        Compute the first derivative of the t_T function
        Args:
            t: float: the time step has shape (batch_size, 1)
        """
        return self.beta_min + t*(self.beta_max - self.beta_min)

    def alpha_t(
            self,
            t: float,
    ):
        """
        Compute the alpha t function
        Args:
            t: float: the time step has shape (batch_size, 1)
        """
        return torch.exp( - 0.5 * self.T_t(t))

    def sigma_t(
            self,
            t: float, 
    ):
        """
        Compute the sigma t function
        Args:
            t: float: the time step has shape (batch_size, 1)
                            """

        return torch.sqrt(1 - self.alpha_t(1 - t)**2)
    
    def mu_t(
            self,
            t: float,
            z_1: torch.Tensor,
    ):
        
        """
        Compute the mu t function
        Args:
            t: float: the time step has shape (batch_size, 1)
            z_1: torch.Tensor: the tensor from the data distribution has shape (batch_size, n_features)
        """
        
        return self.alpha_t(1 - t) * z_1

    def psi_t_conditional_fun(
            self,
            z: torch.Tensor,
            z_1: torch.Tensor,
            t: float,
    ) -> torch.Tensor:
        """
        Compute the conditional flow \psi_t(\vz | \vz_1) with arguments \vz, \vz_1 and t
        Args:
            z: torch.Tensor: the input tensor sampled from the base distribution, has shape (batch_size, n_features)
            z_1: torch.Tensor: the tensor from the data distribution, has shape (batch_size, n_features)
            t: float: the time step, has shape (batch_size, 1)
        """
        assert z.shape == z_1.shape, f"z shape {z.shape} does not match z_1 shape {z_1.shape}"

        if len(t.shape) == 2 and len(z_1.shape) == 3:
            t = t.unsqueeze(-1)
        
        tr = t * (1 - self.epsilon_for_t) # make sure that t is in [0, 1 - epsilon_for_t]

        z_t = self.sigma_t(tr) * z + self.mu_t(tr, z_1)

        return z_t
    
    
    def __call__(self,
                 vector_field_prediction: torch.Tensor,
                 z_0: torch.Tensor,
                 z_1: torch.Tensor,
                 t: float,

    ) -> torch.Tensor:
        """
        Implement the computation of the conditional flow matching loss
        Args:
            vector_field_prediction: torch.Tensor: here the prediction for the noise epsilon(sigma_t*z_0 + mu_t)
            z_0: torch.Tensor: the input tensor sampled from the base distribution 
            z_1: torch.Tensor: the tensor from the data distribution
            t: float: the time step
        """

        target = z_0

        loss = torch.mean((vector_field_prediction - target)**2)

        return loss
    
    def model_prediction_to_vector_field(
            self,
            model: torch.nn.Module,
            z: torch.Tensor,
            x: torch.Tensor,
            t: float,
    ):
        """Compute the vector field from the model that outputs the noise prediction
        Args:
            model: torch.nn.Module: the model that predicts the noise
            z: torch.Tensor: the input tensor
            x: torch.Tensor: the tensor from the data distribution
            t: float: the time step
        """
        if len(z.shape) == 0:
                    t = t.unsqueeze(0)
                    t = t.repeat(z.shape[0], 1)


        tr = t * (1 - self.epsilon_for_t) # make sure that t is in [0, 1 - epsilon_for_t]

        score = model(z, x, tr) / self.sigma_t(tr)

        u_t = - self.T_t_prime(1-tr)/2 * (score - z)

        return u_t

