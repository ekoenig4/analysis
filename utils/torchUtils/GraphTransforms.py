import torch
import torch_geometric 
from torch_geometric.data import Data
from .gnn import useGPU

class Transform:
    def __init__(self,*args):
        self.transforms = args
    def append(self, transform):
        self.transforms.append(transform)
    def __call__(self,graph):
        for transform in self.transforms: 
            graph = transform(graph)
        return graph

class to_uptri_graph:
    def __call__(self,graph):
        edge_index, edge_attr, edge_y = graph.edge_index, graph.edge_attr, graph.edge_y
        uptri = edge_index[0] < edge_index[1]
        edge_index = torch.stack([edge_index[0][uptri], edge_index[1][uptri]])
        edge_attr = edge_attr[uptri]
        edge_y = edge_y[uptri]
        return Data(x=graph.x, y=graph.y, edge_index=edge_index, edge_attr=edge_attr, edge_y=edge_y)

class to_numpy:
    def __call__(self,graph):
        return Data(x=graph.x.numpy(),y=graph.y.numpy(),edge_index=graph.edge_index.numpy(),edge_attr=graph.edge_attr.numpy(),edge_y=graph.edge_y.numpy())

class to_long:
    def __init__(self,precision=1e6):
        self.precision = precision
    def __call__(self,graph):
        return Data(x=(self.precision*graph.x).long(),y=graph.y.long(),edge_index=graph.edge_index.long(),edge_attr=(self.precision*graph.edge_attr).long(),edge_y=graph.edge_y.long())

class to_gpu:
    def __init__(self):
        self._to_gpu = torch_geometric.transforms.ToDevice('cuda:0')
    def __call__(self,graph):
        if useGPU: return self._to_gpu(graph)
        return graph

class use_features:
    def __init__(self, node_mask=[], edge_mask=[]):
        self.node_mask = torch.LongTensor(node_mask)
        self.edge_mask = torch.LongTensor(edge_mask)
    def __call__(self,graph):
        x, edge_attr = graph.x, graph.edge_attr 
        if any(self.node_mask):
            x_mask = (torch.arange(x.shape[1])[...,None] == self.node_mask).any(-1)
            x = x[:,x_mask]
        if any(self.edge_mask):
            e_mask = (torch.arange(edge_attr.shape[1])[...,None] == self.edge_mask).any(-1)
            edge_attr = edge_attr[:,e_mask]
        return Data(x=x, y=graph.y, edge_index=graph.edge_index, edge_attr=edge_attr, edge_y=graph.edge_y)

class mask_features:
    def __init__(self, node_mask=[], edge_mask=[]):
        self.node_mask = torch.LongTensor(node_mask)
        self.edge_mask = torch.LongTensor(edge_mask)
    def __call__(self,graph):
        x, edge_attr = graph.x, graph.edge_attr 
        if any(self.node_mask):
            x_mask = ~(torch.arange(x.shape[1])[...,None] == self.node_mask).any(-1)
            x = x[:,x_mask]
        if any(self.edge_mask):
            e_mask = ~(torch.arange(edge_attr.shape[1])[...,None] == self.edge_mask).any(-1)
            edge_attr = edge_attr[:,e_mask]
        return Data(x=x, y=graph.y, edge_index=graph.edge_index, edge_attr=edge_attr, edge_y=graph.edge_y)
            
    
