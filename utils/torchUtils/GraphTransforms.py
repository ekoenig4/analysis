import torch
import torch_geometric 
from torch_geometric.data import Data
from .gnn import config
from torch_geometric.transforms import BaseTransform
from torch import Tensor

debug = dict(transform=0)

class Transform(BaseTransform):
    def __init__(self,*args):
        self.transforms = list(args)
    def append(self, transform):
        self.transforms.append(transform)
    def insert(self, transform):
        self.transforms.insert(0, transform)
    def __call__(self,graph):
        for transform in self.transforms: 
            graph = transform(graph)
        return graph

class to_uptri_graph(BaseTransform):
    def __call__(self,graph):
        edge_index, edge_attr, edge_y, edge_id = graph.edge_index, graph.edge_attr, graph.edge_y, graph.edge_id
        uptri = edge_index[0] < edge_index[1]
        graph.edge_index = torch.stack([edge_index[0][uptri], edge_index[1][uptri]])
        graph.edge_attr = edge_attr[uptri]
        graph.edge_y = edge_y[uptri]
        graph.edge_id = edge_id[uptri]
        return graph

class to_numpy(BaseTransform):
    def __call__(self,graph):
        for key,value in vars(graph).items():
            setattr(graph, key, value.numpy())
        return graph

class to_long(BaseTransform):
    def __init__(self,precision=1e6):
        self.precision = precision
    def __call__(self,graph):
        return Data(x=(self.precision*graph.x).long(),y=graph.y.long(),edge_index=graph.edge_index.long(),edge_attr=(self.precision*graph.edge_attr).long(),edge_y=graph.edge_y.long())

class to_gpu(BaseTransform):
    def __init__(self):
        self._to_gpu = torch_geometric.transforms.ToDevice('cuda:0')
    def __call__(self,graph):
        if config.useGPU: return self._to_gpu(graph)
        return graph
    
class remove_self_loops(BaseTransform):
    def __call__(self, graph):
        row,col = graph.edge_index 
        
        self_loop_mask = row != col
        
        graph.edge_index = graph.edge_index[:,self_loop_mask]
        graph.edge_attr = graph.edge_attr[self_loop_mask]
        graph.edge_y = graph.edge_y[self_loop_mask]
        return graph
    
class scale_graph(BaseTransform):
    def __init__(self, node_scaler, edge_scaler, type='normalize'):
        self.node_scaler = node_scaler
        self.edge_scaler = edge_scaler
        self.type = type
    def __call__(self,graph):
        graph.x = self.node_scaler.transform(graph.x, self.type)
        graph.edge_attr = self.edge_scaler.transform(graph.edge_attr, self.type)
        return graph
    
class mask_graph(BaseTransform):
    def __init__(self,node_features, node_mask, edge_features, edge_mask):
        if not any(node_mask): node_mask = node_features
        if not any(edge_mask): edge_mask = edge_features
        
        self.node_features = [ feature for feature in node_features if feature in node_mask ]
        self.edge_features = [ feature for feature in edge_features if feature in edge_mask ]
        
        self.node_mask = torch.LongTensor(list(map(node_features.index,self.node_features)))
        self.edge_mask = torch.LongTensor(list(map(edge_features.index,self.edge_features)))
    def __call__(self, graph):
        graph.x = graph.x[:,self.node_mask]
        graph.edge_attr = graph.edge_attr[:,self.edge_mask]
        return graph

class cluster_y(BaseTransform):
    def __call__(self, data : Data) -> Data:
        data.cluster_y = (data.node_id + 3) // 4
        return data
class HyperEdgeY(BaseTransform):
    def __init__(self, permutations=False):
        self.permutations = permutations
    def __call__(self, data : Data) -> Data:
        combs = torch.combinations(torch.arange(data.num_nodes), 4)
        if self.permutations:
            combs = torch.cat([combs, combs[:,[1,0,2,3]],combs[:,[2,0,1,3]],combs[:,[3,0,1,2]]])
        
        data.hyper_edge_index = combs.T
        data.hyper_edge_attr = torch.zeros(data.hyper_edge_index.shape[1],1)

        cluster_y = (data.node_id + 3) // 4
        hyper_edge_y = cluster_y[data.hyper_edge_index.T]
        data.hyper_edge_y = 1*(hyper_edge_y == hyper_edge_y[:,0,None]).all(dim=-1)

        return data

class use_features(BaseTransform):
    def __init__(self, node_mask=[], edge_mask=[]):
        self.node_mask = torch.LongTensor(node_mask)
        self.edge_mask = torch.LongTensor(edge_mask)
    def __call__(self,graph):
        x, edge_attr = graph.x, graph.edge_attr 
        if any(self.node_mask):
            x_mask = (torch.arange(x.shape[1])[...,None] == self.node_mask).any(-1)
            graph.x = x[:,x_mask]
        if any(self.edge_mask):
            e_mask = (torch.arange(edge_attr.shape[1])[...,None] == self.edge_mask).any(-1)
            graph.edge_attr = edge_attr[:,e_mask]
        return graph

class mask_features(BaseTransform):
    def __init__(self, node_mask=[], edge_mask=[]):
        self.node_mask = torch.LongTensor(node_mask)
        self.edge_mask = torch.LongTensor(edge_mask)
    def __call__(self,graph):
        x, edge_attr = graph.x, graph.edge_attr 
        if any(self.node_mask):
            x_mask = ~(torch.arange(x.shape[1])[...,None] == self.node_mask).any(-1)
            graph.x = x[:,x_mask]
        if any(self.edge_mask):
            e_mask = ~(torch.arange(edge_attr.shape[1])[...,None] == self.edge_mask).any(-1)
            graph.edge_attr = edge_attr[:,e_mask]
        return graph
        
class MotherNode(BaseTransform):
    def __init__(self, level, id=-1, pairs=[], connects=-1):
        self.level = level
        self.id = id
        self.id1, self.id2 = pairs
        self.connects = connects
        
    def __call__(self, data: Data) -> Data:
        num_nodes, (row, col) = data.num_nodes, data.edge_index
        edge_type = data.get('edge_type', torch.zeros_like(row))
        node_type = data.get('node_type', torch.zeros(num_nodes))

        connected_nodes = node_type == self.connects
        arange = torch.arange(num_nodes, device=row.device)[connected_nodes]
        num_connected = len(arange)
        
        full = row.new_full((num_connected, ), num_nodes)
        row = torch.cat([row, arange, full], dim=0)
        col = torch.cat([col, full, arange], dim=0)
        edge_index = torch.stack([row, col], dim=0)

        new_type = edge_type.new_full((num_connected, ), self.level)
        edge_type = torch.cat([edge_type, new_type, new_type], dim=0)
        
        new_type = node_type.new_full((1, ), self.level)
        node_type = torch.cat([node_type, new_type], dim=0)

        for key, value in data.items():
            if key == 'edge_index' or key == 'edge_type':
                continue

            if isinstance(value, Tensor):
                dim = data.__cat_dim__(key, value)
                size = list(value.size())

                fill_value = None
                if data.is_edge_attr(key):
                    size[dim] = 2 * num_connected
                    fill_value = 0.
                elif data.is_node_attr(key):
                    size[dim] = 1
                    fill_value = 0.
                elif key == 'node_id':
                    size[dim] = 1
                    fill_value = self.id

                if fill_value is not None:
                    new_value = value.new_full(size, fill_value)
                    data[key] = torch.cat([value, new_value], dim=dim)
        

        # decays = (((data.node_id[row] == self.id) & (data.node_id[col] == self.id1)) | ((data.node_id[row] == self.id1) & (data.node_id[col] == self.id)) |
        #           ((data.node_id[row] == self.id) & (data.node_id[col] == self.id2)) | ((data.node_id[row] == self.id2) & (data.node_id[col] == self.id)))
        # data.edge_y = torch.where(decays, 1, data.edge_y)
        data.edge_index = edge_index
        data.edge_type = edge_type
        data.node_type = node_type
        if 'num_nodes' in data:
            data.num_nodes = data.num_nodes + 1

        return data

from torch_scatter import scatter_max , scatter_min

def k_min_neighbors(d, edge_index, n_neighbor=1, remove_self=False):
    d = d.clone()
    fill_value = d.max()+1
    used_edges = edge_index[0] == edge_index[1]
    d[used_edges] = fill_value

    for n in range(n_neighbor):
        edge_n = scatter_min(d, edge_index[0], dim=0)[1]
        edge_n = edge_n[~used_edges[edge_n]]
        used_edges[edge_n] = True
        d[edge_n] = fill_value

    if remove_self: used_edges[edge_index[0] == edge_index[1]] = False
    return used_edges

def k_max_neighbors(d, edge_index, n_neighbor=1, remove_self=False):
    d = d.clone()
    fill_value = d.min()-1
    used_edges = edge_index[0] == edge_index[1]
    d[used_edges] = fill_value

    for n in range(n_neighbor):
        edge_n = scatter_max(d, edge_index[0], dim=0)[1]
        edge_n = edge_n[~used_edges[edge_n]]
        used_edges[edge_n] = True
        d[edge_n] = fill_value

    if remove_self: used_edges[edge_index[0] == edge_index[1]] = False
    return used_edges
class min_edge_neighbor(BaseTransform):
    def __init__(self, n_neighbor=4, features=0, function=lambda f : f**2):
        self.n_neighbor = n_neighbor
        self.features = features
        self.function = function

    def __call__(self, data : Data) -> Data:
        features = data.edge_attr[:,self.features].clone()
        if self.function is not None: features = self.function(features)
        data.edge_d = features
        data.edge_mask = k_min_neighbors(features, data.edge_index, self.n_neighbor)
        return data


class max_edge_neighbor(BaseTransform):
    def __init__(self, n_neighbor=4, features=2, function=None):
        self.n_neighbor = n_neighbor
        self.features = features
        self.function = function

    def __call__(self, data : Data) -> Data:
        features = data.edge_attr[:,self.features].clone()
        if self.function is not None: features = self.function(features)
        data.edge_d = features
        data.edge_mask = k_max_neighbors(features, data.edge_index, self.n_neighbor)
        return data
