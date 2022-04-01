
import torch.nn.functional as F
from .gnn import select_top_edges

def std_loss(model, node_o, edge_o, batch, **kwargs):
    node_loss = F.nll_loss(node_o, batch.y, model.node_weights)
    edge_loss = F.nll_loss(
        edge_o, batch.edge_y, model.edge_weights)
    loss = model.type_weights[0]*node_loss + model.type_weights[1]*edge_loss
    return loss

def std_loss_plus_top4edges(model, node_o, edge_o, batch, selected_edges=None,**kwargs):
    node_loss = F.nll_loss(node_o, batch.y, model.node_weights)
    edge_loss = F.nll_loss(
        edge_o, batch.edge_y, model.edge_weights)
    
    if selected_edges is None:
        selected_edges = select_top_edges(edge_o, batch)
        
    edge_loss += F.nll_loss(edge_o[selected_edges], batch.edge_y[selected_edges])
    loss = model.type_weights[0]*node_loss + model.type_weights[1]*edge_loss
    return loss

def std_loss_top4edges(model, node_o, edge_o, batch, selected_edges=None, **kwargs):
    node_loss = F.nll_loss(node_o, batch.y, model.node_weights)
    
    if selected_edges is None:
        selected_edges = select_top_edges(edge_o, batch)
        
    edge_loss = F.nll_loss(edge_o[selected_edges], batch.edge_y[selected_edges])
    loss = model.type_weights[0]*node_loss + model.type_weights[1]*edge_loss
    return loss
    
local_functions = locals()
lossMap = {
    loss : local_functions[loss]
    for loss in filter(lambda f : 'loss' in f,dir())
}