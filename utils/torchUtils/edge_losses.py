
import torch.nn.functional as F
from torch import Tensor
from .gnn import select_top_edges

def std_loss(model, edge_o, batch, **kwargs):
    edge_loss = F.nll_loss(
        edge_o, batch.edge_y, model.edge_weights)
    return model.type_weights[1]*edge_loss

mismatch_weights = Tensor([4.2755, 1.1581, 1.0000, 7.2572])
def mismatched_bjet_loss(model, edge_o, batch, **kwargs):
    weights = mismatch_weights.to(model.device)

    ni, nj = batch.edge_index
    ni_y, nj_y = batch.y[ni], batch.y[nj]

    n_same = ni_y == nj_y
    n_diff = ni_y ^  nj_y

    def _get_loss(mask):
        if mask.sum() == 0: return 0
        return F.nll_loss(edge_o[mask], batch.edge_y[mask])

    def _get_weighted_loss(mask, weights):
        if mask.sum() == 0: return 0
        return F.nll_loss(edge_o[mask], batch.edge_y[mask], weights)

    loss_00 = weights[0]*_get_loss((ni_y == 0) & n_same)
    loss_01 = weights[1]*_get_loss(n_diff)
    loss_11 = _get_weighted_loss((ni_y == 1) & n_same, weights[2:])
    return model.type_weights[1] * (loss_00 + loss_01 + loss_11)

def std_loss_plus_top4edges(model, edge_o, batch, selected_edges=None,**kwargs):
    edge_loss = F.nll_loss(
        edge_o, batch.edge_y, model.edge_weights)
    
    if selected_edges is None:
        selected_edges = select_top_edges(edge_o, batch)
        
    edge_loss += F.nll_loss(edge_o[selected_edges], batch.edge_y[selected_edges])
    return model.type_weights[1]*edge_loss

def std_loss_top4edges(model, edge_o, batch, selected_edges=None, **kwargs):
    if selected_edges is None:
        selected_edges = select_top_edges(edge_o, batch)
        
    edge_loss = F.nll_loss(edge_o[selected_edges], batch.edge_y[selected_edges])
    return model.type_weights[1]*edge_loss
   
__all__ = [
    "std_loss", "std_loss_plus_top4edges", "std_loss_top4edges", "mismatched_bjet_loss"
]