import torch.nn.functional as F
from torch import Tensor

hyper_edge_weights = Tensor([ 1.0000, 72.7041])
hyper_edge_type_weight = 0.5343

def std_loss(model, hyper_edge_o, batch):
    hyper_edge_loss = F.nll_loss(hyper_edge_o, batch.hyper_edge_y, hyper_edge_weights.to(model.device))
    return hyper_edge_type_weight * hyper_edge_loss