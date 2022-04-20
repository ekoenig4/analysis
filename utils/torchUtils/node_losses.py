
import torch.nn.functional as F

def std_loss(model, node_o, batch, **kwargs):
    node_loss = F.nll_loss(node_o, batch.y, model.node_weights)
    return model.type_weights[0]*node_loss

__all__ = [ 
    "std_loss"
]