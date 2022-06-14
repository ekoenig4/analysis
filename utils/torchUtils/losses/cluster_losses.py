from torch import Tensor
import torch.nn.functional as F

cluster_weights = Tensor([1.2777, 1.0000, 1.0724])

def std_loss(model, cluster_o, batch, **kwargs):
    cluster_loss = F.nll_loss(cluster_o, batch.cluster_y)
    return cluster_loss