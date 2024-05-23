import torch 


def add_l2_loss_nll_loss(loss_fun: callable, lambda_value: float = 0.0):
    """
    Augments a nll loss function by adding an l2 loss between the prediction and the target
    Args:
        loss_fun: callable: the nll loss function
        lambda_value: float: the weight of the l2 loss used in a convex combination with the nll loss. Default is 0.0, i.e only nll loss is used
    Returns:
        callable: the augmented loss function
    """

    def augmented_loss(pred, target):
        nll = loss_fun(pred, target)
        l2 = torch.nn.MSELoss()(pred[0], target)
        
        final_loss = nll * (1 - lambda_value) + l2 * lambda_value

        return final_loss
    
    return augmented_loss