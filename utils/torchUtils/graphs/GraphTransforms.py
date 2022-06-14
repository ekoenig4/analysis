import torch
import torch_geometric 
from torch_geometric.data import Data
from ..gnn import config, k_max_neighbors, k_min_neighbors, sample_center, mask_graph_edges, mask_undirected, mask_graph_nodes
from torch import Tensor
from torch_scatter import scatter_max

debug = dict(transform=0)

class BaseTransform(torch_geometric.transforms.BaseTransform):
    def __init__(self, **kwargs):
        self.hparams = kwargs
    def __call__(self, data : Data) -> Data:
        if data.num_nodes == 0 or data.num_edges == 0: return data
        return self.transform(data)

class Transform(BaseTransform):
    def __init__(self,*args):
        super().__init__()
        self.transforms = list(args)
        for transform in self: self.hparams.update(transform.hparams)
    def append(self, transform):
        self.transforms.append(transform)
        self.hparams.update(transform.hparams)
    def insert(self, transform):
        self.transforms.insert(0, transform)
        self.hparams.update(transform.hparams)
    def __iter__(self): return iter(self.transforms)
    def transform(self,graph):
        for transform in self: 
            graph = transform(graph)
        return graph
    def __add__(self, other):   
        return Transform(*self.transforms, other)

class to_uptri_graph(BaseTransform):
    def transform(self,graph):
        edge_index = graph.edge_index
        uptri = edge_index[0] < edge_index[1]
        graph.edge_index = edge_index[:, uptri]
        for key, value in graph.items():
            if value.shape[0] == uptri.shape[0]:
                graph[key] = value[uptri]
        return graph

class to_numpy(BaseTransform):
    def transform(self,graph):
        for key,value in vars(graph).items():
            setattr(graph, key, value.numpy())
        return graph

class to_long(BaseTransform):
    def __init__(self,precision=1e6):
        super().__init__(precision=precision)
        self.precision = precision
    def transform(self,graph):
        return Data(x=(self.precision*graph.x).long(),y=graph.y.long(),edge_index=graph.edge_index.long(),edge_attr=(self.precision*graph.edge_attr).long(),edge_y=graph.edge_y.long())

class to_gpu(BaseTransform):
    def __init__(self):
        super().__init__()
        self._to_gpu = torch_geometric.transforms.ToDevice('cuda:0')
    def transform(self,graph):
        if config.useGPU: return self._to_gpu(graph)
        return graph
    
class remove_self_loops(BaseTransform):
    def transform(self, graph):
        row,col = graph.edge_index 
        
        self_loop_mask = row != col
        
        graph.edge_index = graph.edge_index[:,self_loop_mask]
        graph.edge_attr = graph.edge_attr[self_loop_mask]
        graph.edge_y = graph.edge_y[self_loop_mask]
        return graph
    
class scale_graph(BaseTransform):
    def __init__(self, node_scaler, edge_scaler, type='normalize'):
        super().__init__()
        self.node_scaler = node_scaler
        self.edge_scaler = edge_scaler
        self.type = type
    def transform(self,graph):
        graph.x = self.node_scaler.transform(graph.x, self.type)
        graph.edge_attr = self.edge_scaler.transform(graph.edge_attr, self.type)
        return graph
    
class mask_graph(BaseTransform):
    def __init__(self,node_features, node_mask, edge_features, edge_mask):
        super().__init__()
        if not any(node_mask): node_mask = node_features
        if not any(edge_mask): edge_mask = edge_features
        
        self.node_features = [ feature for feature in node_features if feature in node_mask ]
        self.edge_features = [ feature for feature in edge_features if feature in edge_mask ]
        
        self.node_mask = torch.LongTensor(list(map(node_features.index,self.node_features)))
        self.edge_mask = torch.LongTensor(list(map(edge_features.index,self.edge_features)))
    def transform(self, graph):
        graph.x = graph.x[:,self.node_mask]
        graph.edge_attr = graph.edge_attr[:,self.edge_mask]
        return graph

class cluster_y(BaseTransform):
    def transform(self, data : Data) -> Data:
        data.cluster_y = (data.node_id + 3) // 4
        return data

class HyperEdgeY(BaseTransform):
    def __init__(self, permutations=False):
        super().__init__(permutations=permutations)
        self.permutations = permutations
    def transform(self, data : Data) -> Data:
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
        super().__init__()
        self.node_mask = torch.LongTensor(node_mask)
        self.edge_mask = torch.LongTensor(edge_mask)
    def transform(self,graph):
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
        super().__init__()
        self.node_mask = torch.LongTensor(node_mask)
        self.edge_mask = torch.LongTensor(edge_mask)
    def transform(self,graph):
        x, edge_attr = graph.x, graph.edge_attr 
        if any(self.node_mask):
            x_mask = ~(torch.arange(x.shape[1])[...,None] == self.node_mask).any(-1)
            graph.x = x[:,x_mask]
        if any(self.edge_mask):
            e_mask = ~(torch.arange(edge_attr.shape[1])[...,None] == self.edge_mask).any(-1)
            graph.edge_attr = edge_attr[:,e_mask]
        return graph
class min_edge_neighbor(BaseTransform):
    def __init__(self, n_neighbor=4, features=0, function=lambda f : (f+0.265)**2, undirected=False):
        super().__init__(n_neighbor=n_neighbor, undirected=undirected)
        self.n_neighbor = n_neighbor
        self.undirected = undirected
        self.features = features
        self.function = function

    def transform(self, data : Data) -> Data:
        features = data.edge_attr[:,self.features].clone()
        if self.function is not None: features = self.function(features)
        data.edge_d = features
        data.edge_mask = k_min_neighbors(features, data.edge_index, self.n_neighbor, undirected=self.undirected)
        return data


class max_edge_neighbor(BaseTransform):
    def __init__(self, n_neighbor=4, features=2, function=None, undirected=False):
        super().__init__(n_neighbor=n_neighbor, undirected=undirected)
        self.n_neighbor = n_neighbor
        self.undirected = undirected
        self.features = features
        self.function = function

    def transform(self, data : Data) -> Data:
        features = data.edge_attr[:,self.features].clone()
        if self.function is not None: features = self.function(features)
        data.edge_d = features
        data.edge_mask = k_max_neighbors(features, data.edge_index, self.n_neighbor, undirected=self.undirected)
        return data

class edge_neighborhood(BaseTransform):
    def __init__(self, features=0, function=lambda f : torch.abs(f+0.265), selection=lambda f : f<0.75, undirected=False):
        super().__init__(undirected=undirected)
        self.undirected = undirected
        self.features = features
        self.function = function
        self.selection = selection
    def transform(self, data : Data) -> Data:
        features = data.edge_attr[:,self.features].clone()
        if self.function is not None: features = self.function(features)
        data.edge_d = features

        data.edge_mask = self.selection(features)
        if self.undirected:
            data.edge_mask = mask_undirected(data.edge_index, data.edge_mask)
        return data

class mask_edges(BaseTransform):
    def transform(self, data : Data) -> Data:
        edge_mask = data.get('edge_mask', None)
        if edge_mask is None: return data
        return mask_graph_edges(data, edge_mask)
class RandomSample(BaseTransform):
    def __init__(self):
        super().__init__(random_sampled=True)
    def transform(self, data : Data) -> Data:
        paired = scatter_max(data.edge_y, data.edge_index[0], dim=0)[0]
        if not torch.any(paired == 1): return data

        pos_nodes = torch.where(paired==1)[0]
        center = pos_nodes[torch.randint(pos_nodes.shape[0], (1,))]

        edge_mask = sample_center(data, center)
        return mask_graph_edges(data, edge_mask)

class SamplePair(BaseTransform):
    def __init__(self, n_pos=1, n_neg=6):
        super().__init__(n_pos=n_pos, n_neg=n_neg)
        self.n_pos = n_pos
        self.n_neg = n_neg
    def transform(self, data : Data) -> Data:
        uptri = data.edge_index[0] < data.edge_index[1]
        pos_edges = torch.where(uptri & (data.edge_y == 1))[0]
        neg_edges = torch.where(uptri & (data.edge_y == 0))[0]
        if pos_edges.shape[0] == 0: pos_edges = neg_edges   

        n_pos = pos_edges.shape[0]
        if n_pos < self.n_pos:
            while pos_edges.shape[0] < self.n_pos:
                pos_edges = torch.cat((pos_edges,pos_edges[torch.randint(n_pos, (1,))]))

        pos_edge = pos_edges[ torch.randperm(pos_edges.shape[0])[:self.n_pos] ]
        neg_edge = neg_edges[ torch.randperm(neg_edges.shape[0])[:self.n_neg] ]
        pos_edge = data.edge_index[:, pos_edge]
        neg_edge = data.edge_index[:, neg_edge]

        data.pos_index = pos_edge.T.reshape(self.n_pos,2,-1)
        data.neg_index = neg_edge.T.reshape(self.n_neg,2,-1)
        return data

class SampleCenter(BaseTransform):
    def __init__(self, n_pairs=5):
        super().__init__(n_pairs=n_pairs)
        self.n_pairs = n_pairs
    def transform(self, data : Data) -> Data:
        arange = torch.arange(data.num_nodes)
        center = arange[torch.randint(data.num_nodes,(1,))]
        # center = arange[3]
        uptri = data.edge_index[0] < data.edge_index[1]
        def _get_connected_edges(edge_idx):
            edge_idx = data.edge_index[:,edge_idx]
            edge_mask = (edge_idx == center).any(dim=0)
            edge_idx = edge_idx[:, edge_mask]
            return edge_idx

        neg_edges = torch.where(uptri & (data.edge_y == 0))[0]
        neg_edges = _get_connected_edges(neg_edges)
        neg_edges = neg_edges[:, torch.randperm(neg_edges.shape[1])[:self.n_pairs]]

        pos_edges = neg_edges[:,0]
        if data.edge_y.sum() > 0:
            pos_edges = torch.where(uptri & (data.edge_y == 1))[0]
            pos_edges = _get_connected_edges(pos_edges)
            if pos_edges.shape[1] == 0:
                pos_edges = torch.where(uptri & (data.edge_y == 1))[0]
                pos_edges = pos_edges[torch.randint(pos_edges.shape[0], (1,))]
                pos_edges = data.edge_index[:, pos_edges]

        data.pos_index = pos_edges.T.reshape(1, 2, -1)
        data.neg_index = neg_edges.T.reshape(self.n_pairs, 2, -1)        
        return data

class Top8Btag(BaseTransform):
    def transform(self, data : Data) -> Data:
        top8_mask = data.x[:,4].argsort(descending=True).argsort() < 8
        edge_mask = top8_mask[data.edge_index].all(dim=0)
        data = mask_graph_edges(data, edge_mask)
        data = mask_graph_nodes(data, top8_mask)
        return data

class SimplePair(BaseTransform):
    def __init__(self, n_pos=1, n_neg=8):
        super().__init__(n_pos=n_pos, n_neg=n_neg)
        self.n_pos = n_pos 
        self.n_neg = n_neg
    def transform(self, data : Data) -> Data:
        uptri = data.edge_index[0] < data.edge_index[1]
        pos_edges = torch.where(uptri & (data.edge_y==1))[0]

        pos_edges = data.edge_index[:,pos_edges]
        neg_edges = torch.where(uptri & (data.edge_y==0))[0]
        neg_edges = data.edge_index[:,neg_edges]

        pos_order = data.x[pos_edges][:,:,1].sum(dim=0).argsort(descending=True)
        pos_edges = pos_edges[:,pos_order]

        if pos_edges.shape[1] >= self.n_pos:
            pos_edge = pos_edges[:,0] 
            pos_edge = pos_edges[:,:self.n_pos].T.reshape(self.n_pos,2,-1)
            pos_mask = (data.edge_index == pos_edge).all(dim=1)
            data.pos_mask_index = 1*pos_mask

        neg_edge = neg_edges[:, torch.randperm(neg_edges.shape[1])[:self.n_neg] ].T.reshape(self.n_neg, 2,-1)
        neg_mask = (data.edge_index == neg_edge).all(dim=1)
        data.neg_mask_index = 1*neg_mask

        return data