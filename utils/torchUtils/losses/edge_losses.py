
import torch.nn.functional as F
from torch import Tensor
import torch
from ..gnn import select_top_edges, attr_undirected

def std_loss(model, edge_o, batch, **kwargs):
    edge_loss = F.nll_loss(
        edge_o, batch.edge_y, model.edge_weights)
    return model.type_weights[1]*edge_loss

def undirected_loss(model, edge_o, batch, **kwargs):
    edge_o = F.log_softmax(attr_undirected(batch.edge_index, edge_o),dim=-1)
    uptri = batch.edge_index[0] < batch.edge_index[1]
    loss = F.nll_loss(edge_o[uptri], batch.edge_y[uptri], model.edge_weights)
    return loss

def auroc_loss(model, edge_o, batch, **kwargs):
    score = torch.exp(edge_o[:,1])

    edge_batch = batch.batch[batch.edge_index[0]]
    n_edges = edge_batch.unique(return_counts=True)[1]
    ptr = torch.cat([torch.zeros(1,).to(model.device),torch.cumsum(n_edges, dim=0)])[:-1].long()
    rows = torch.cat([ torch.repeat_interleave(torch.arange(n1).to(model.device),n1) + n0 for n0,n1 in zip(ptr,n_edges) ])
    cols = torch.cat([ torch.repeat_interleave(torch.arange(n1)[None].to(model.device),n1, dim=0).reshape(-1) + n0 for n0,n1 in zip(ptr,n_edges) ])
    row_pos = batch.edge_y[rows] == 1
    col_neg = batch.edge_y[cols] == 0
    rows = rows[row_pos & col_neg]
    cols = cols[row_pos & col_neg]

    norm = torch.repeat_interleave(n_edges, n_edges)[rows]
    s = (1-score[rows]+score[cols])/norm

    n_graphs = batch.num_graphs if hasattr(batch,'num_graphs') else 1
    loss = torch.sum(F.relu(s)**2)/n_graphs
    return loss

def std_auroc_loss(model, edge_o, batch, **kwargs):
    n_graphs = batch.num_graphs if hasattr(batch,'num_graphs') else 1
    l1 = std_loss(model, edge_o, batch)
    l2 = auroc_loss(model, edge_o, batch)
    return l1+l2

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
   
# __all__ = [
#     "std_loss", "std_loss_plus_top4edges", "std_loss_top4edges", "mismatched_bjet_loss", "ranked_loss"
# ]