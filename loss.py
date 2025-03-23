import torch

def priv_loss(output, target):
    # TODO: add privQA considerations
    loss = torch.mean((output - target)**2)
    return loss
