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

def convex_combine(loss_fun_lis: list[callable], lambda_values: list[float]):
    """
    Augments a list of loss functions by taking a convex combination of them
    Args:
        loss_fun_lis: list[callable]: the list of loss functions
        lambda_values: list[float]: the weights of the loss functions used in the convex combination
    Returns:
        callable: the augmented loss function
    """

    assert len(loss_fun_lis) == len(lambda_values), "The number of loss functions should match the number of lambda values"

    if not sum(lambda_values) == 1:
        print("The lambda values do not sum to 1, normalizing them")
        lambda_values = [i / sum(lambda_values) for i in lambda_values]

    def augmented_loss(pred, target):
        loss = 0
        for i, loss_fun in enumerate(loss_fun_lis):
            loss += loss_fun(pred, target) * lambda_values[i]

        return loss
    
    return augmented_loss