
import torch.nn.functional as F
import torch

from .gnn import to_tensor

def ntop_edge_loss(edge_o, edge_y, n_edges, n_top=4):
    from torch_scatter import scatter_min
    def get_top_edge_loss(edge_o, edge_y, n_edges):
        idx = to_tensor(torch.arange(len(n_edges)))
        edge_batch = torch.repeat_interleave(idx,n_edges)
        _,edge_arg = scatter_min(edge_o[:,0],edge_batch)
        
        min_edge_o = edge_o[edge_arg]
        min_edge_y = edge_y[edge_arg]
        idx = to_tensor(torch.arange(len(edge_o[:,0])))
        remaining = ~(idx[...,None] == edge_arg).any(-1)
        
        loss = F.nll_loss(min_edge_o, min_edge_y)
        
        return loss, (edge_o[remaining], edge_y[remaining], n_edges-1)
    remaining_edges = (edge_o,edge_y,n_edges)
    loss = 0
    for i in range(n_top):
        top_loss, remaining_edges = get_top_edge_loss(*remaining_edges)
        loss += top_loss
    return loss

def std_loss(model, node_o, edge_o, batch):
    node_loss = F.nll_loss(node_o, batch.y, model.node_weights)
    edge_loss = F.nll_loss(
        edge_o, batch.edge_y, model.edge_weights)
    loss = model.type_weights[0]*node_loss + model.type_weights[1]*edge_loss
    return loss

def std_loss_plus_top4edges(model, node_o, edge_o, batch):
    node_loss = F.nll_loss(node_o, batch.y, model.node_weights)
    edge_loss = F.nll_loss(
        edge_o, batch.edge_y, model.edge_weights)
    
    n_nodes = batch.ptr[1:]-batch.ptr[:-1]
    n_edges = n_nodes*(n_nodes-1)//2
    edge_loss += ntop_edge_loss(edge_o, batch.edge_y, n_edges)
    loss = model.type_weights[0]*node_loss + model.type_weights[1]*edge_loss
    return loss
    