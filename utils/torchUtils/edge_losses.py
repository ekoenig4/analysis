
import torch.nn.functional as F
from .gnn import select_top_edges

def std_loss(model, edge_o, batch, **kwargs):
    edge_loss = F.nll_loss(
        edge_o, batch.edge_y, model.edge_weights)
    return model.type_weights[1]*edge_loss

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
    "std_loss", "std_loss_plus_top4edges", "std_loss_top4edges"
]