
import torch.nn.functional as F
from torch import Tensor
import torch
from torch_scatter import scatter_max, scatter_min

def mass_spread_loss(batch, quad_mask=None):
    inv_m = batch.edge_attr[quad_mask,0]

    n_edges = (batch.ptr[1:]-batch.ptr[:-1])**2
    n_graph = batch.num_graphs
    batch_edge = torch.repeat_interleave(torch.arange(n_graph), n_edges)
    batch_edge = batch_edge[quad_mask]

    diff_m = torch.abs(scatter_max(inv_m, batch_edge, dim=0)[0] - scatter_min(inv_m, batch_edge, dim=0)[0])
    return diff_m