import torch
from torch.utils.data import random_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import sklearn.metrics as metrics
import multiprocessing

from ..selectUtils import *
from ..utils import *

class config:
    useGPU = True
    useGPU = useGPU and torch.cuda.is_available()
    device = 'cuda:0' if useGPU else 'cpu'
    
    ncpu = multiprocessing.cpu_count()
    ngpu = 1*useGPU
    
    @staticmethod
    def set_gpu(flag):
        config.useGPU = flag and torch.cuda.is_available()
        config.ngpu = 1*config.useGPU
        config.device = 'cuda:0' if config.useGPU else 'cpu'

def to_tensor(tensor):
    if not torch.is_tensor(tensor):
        tensor = torch.Tensor(tensor)
    return tensor.to(device=config.device)


def get_uptri(edge_index, edge_attr, return_index=False):
    uptri = edge_index[0] < edge_index[1]
    edge_index = torch.stack([edge_index[0][uptri], edge_index[1][uptri]])
    edge_attr = edge_attr[uptri]
    if return_index:
        return edge_attr, edge_index
    return edge_attr


def train_test_split(dataset, test_split):
    size = len(dataset)
    train_size = int(size*(1-test_split))
    test_size = size - train_size
    return random_split(dataset, [train_size, test_size])


def graph_pred(model, g):
    edge_pred = model.predict(g)
    if type(edge_pred) is tuple:
        _, edge_pred = edge_pred

    g_pred = Data(x=g.x, edge_index=g.edge_index,
                  edge_attr=g.edge_attr, y=g.y, edge_y=g.edge_y, edge_pred=edge_pred)
    return g_pred


def get_wp(wp_fpr, fpr, tpr, thresholds):
    wp_index = np.where(fpr > wp_fpr)[0][0]
    wp_tpr = tpr[wp_index]
    wp_threshold = thresholds[wp_index]
    return np.array([wp_fpr, wp_tpr, wp_threshold])

def predict_dataset_edges(model, dataset, batch_size=50):
    if type(dataset) is not DataLoader:
        dataset = DataLoader(dataset, batch_size=batch_size, num_workers=4)
    edge_scores = torch.cat([model.predict_edges(data) for data in dataset])
    return edge_scores.numpy()

def predict_dataset_nodes(model, dataset, batch_size=50):
    if type(dataset) is not DataLoader:
        dataset = DataLoader(dataset, batch_size=batch_size, num_workers=4)
    edge_scores = torch.cat([model.predict_nodes(data) for data in dataset])
    return edge_scores.numpy()

def mask_graph_nodes(data, node_mask):
    for key,value in data.items():
        if value.shape[0] == node_mask.shape[0]:
            data[key] = value[node_mask]

    arange = torch.arange(data.num_nodes)
    row = torch.repeat_interleave(arange, data.num_nodes)
    col = torch.cat([arange]*data.num_nodes)
    data.edge_index = torch.stack([row,col])

    return data

def mask_graph_edges(data, edge_mask):
    data.edge_index = data.edge_index[:,edge_mask]
    for key,value in data.items():
        if value.shape[0] == edge_mask.shape[0]:
            data[key] = value[edge_mask]
    return data

def sample_center(data, center):
    edge_mask = (data.edge_index == center).any(dim=0)
    if hasattr(data, 'edge_mask'):
        edge_mask = edge_mask & data.edge_mask
    return edge_mask

def k_max_accuracy(edge_o, graph, k=1):
    edge_mask = k_max_neighbors(edge_o, graph.edge_index, k, remove_self=True)

    n_higgs = (graph.edge_id.unique(return_counts=True)[1][1:]).sum()
    n_edges = (graph.edge_id[edge_mask].unique(return_counts=True)[1][1:]).sum()
    return n_edges/n_higgs

def top_accuracy(score, batch):
    edge_batch = batch.get('batch', torch.zeros(batch.num_nodes))[batch.edge_index[0]]
    pred_edges = scatter_max(score, edge_batch, dim=0)[1]
    true_edges = scatter_max(batch.edge_y, edge_batch, dim=0)[1]
    return batch.edge_y[pred_edges].sum() / batch.edge_y[true_edges].sum()


class ROCMetric:
    def __init__(self, true, pred):
        self.true = ak.flatten(true,axis=None).to_numpy()
        self.pred = ak.flatten(pred,axis=None).to_numpy()
        self.fpr, self.tpr, self.thresholds = metrics.roc_curve(
            self.true, self.pred)
        self.auc = metrics.auc(self.fpr, self.tpr)

    def get_wps(self, fpr_wps=[0.2, 0.1, 0.05]):
        self.wps = np.stack(
            [get_wp(fpr_wp, self.fpr, self.tpr, self.thresholds) for fpr_wp in fpr_wps])
        return self.wps

    def get_values(self): return self.fpr, self.tpr, self.auc


def get_model_roc(model, dataloader, batch_size=50):
    if type(dataloader) is not DataLoader:
        dataloader = DataLoader(dataloader, batch_size=batch_size, num_workers=4)

    node_true = torch.cat([data.y for data in dataloader]).numpy()
    node_pred = predict_dataset_nodes(model, dataloader)

    edge_true = torch.cat([data.edge_y for data in dataloader]).numpy()
    edge_pred = predict_dataset_edges(model, dataloader)

    node_metrics = ROCMetric(node_true, node_pred)
    edge_metrics = ROCMetric(edge_true, edge_pred)

    return node_metrics, edge_metrics



from torch import Tensor, LongTensor
from torch_scatter import scatter_max

@torch.jit.script
def _next_edge(edge_o : Tensor, edge_index : LongTensor, edge_id : LongTensor, graph_id : LongTensor):
    edge_arg_max = scatter_max(edge_o[:,1], graph_id)[1]
    edge_index_max = edge_index[:,edge_arg_max]
    edge_id_max = edge_id[edge_arg_max]
    used_edges = (edge_index[0,...,None] == edge_index_max.reshape(-1)).any(-1) | (edge_index[1,...,None] == edge_index_max.reshape(-1)).any(-1)
    
    edge_o = edge_o[~used_edges]
    edge_index = edge_index[:,~used_edges]
    edge_id = edge_id[~used_edges]
    graph_id = graph_id[~used_edges]
    return edge_id_max,(edge_o, edge_index, edge_id, graph_id)

@torch.jit.script
def _select_top_edges(edge_o : Tensor, edge_index : LongTensor, edge_id : LongTensor, graph_id : LongTensor, n_top : int):
    next_args = (edge_o, edge_index, edge_id, graph_id)
    selected_edges = []
    for i in range(n_top):
        edge_id_max, next_args = _next_edge(*next_args)
        selected_edges.append(edge_id_max)
    selected_edges = torch.cat(selected_edges)
    return selected_edges
    
def select_top_edges(edge_o, batch, n_top=4):
    n_nodes = batch.ptr[1:]-batch.ptr[:-1]
    n_edges = n_nodes*(n_nodes-1)//2
    
    edge_id = torch.arange(n_edges.sum()).to(device=config.device)
    graph_id = torch.repeat_interleave(torch.arange(len(n_edges)).to(device=config.device),n_edges).to(device=config.device)
    return _select_top_edges(edge_o, batch.edge_index, edge_id, graph_id, n_top)


from torch_scatter import scatter_max , scatter_min

def mask_undirected(edge_index, edge_mask):
    dense = torch.sparse_coo_tensor(edge_index, 1*edge_mask).to_dense() == 1
    dense = (dense | dense.T)
    return dense[edge_index[0], edge_index[1]]

_attr_merge = dict(
    avg=lambda di, dj: (di+dj)/2,
    max=lambda di, dj:torch.cat([di.unsqueeze(-1), dj.unsqueeze(-1)],dim=-1).max(dim=-1)[0]
)    

def attr_undirected(edge_index, edge_attr, merge='avg'):
    dense = torch.sparse_coo_tensor(edge_index, edge_attr).to_dense()
    di, dj = dense[edge_index[[0,1]], edge_index[[1,0]]]
    dense = _attr_merge[merge](di, dj)
    return dense

def k_min_neighbors(d, edge_index, n_neighbor=1, remove_self=False, return_steps=False, undirected=False):
    d = d.clone()
    fill_value = d.max()+1
    self_edges = edge_index[0] == edge_index[1]
    d[self_edges] = fill_value

    used_edges = torch.zeros_like(self_edges) == 1
    steps = []

    for n in range(n_neighbor):
        edge_n = scatter_min(d, edge_index[0], dim=0)[1]
        edge_n = edge_n[~used_edges[edge_n]]
        used_edges[edge_n] = True
        steps.append( used_edges.clone() )
        d[edge_n] = fill_value

    if return_steps:
        if not remove_self:
            steps = [ step | self_edges for step in steps ]
        if undirected:
            steps = [ mask_undirected(edge_index, step) for step in steps ]
        return steps

    if not remove_self: 
        used_edges = used_edges | self_edges
    if undirected:
        used_edges = mask_undirected(edge_index, used_edges)
    return used_edges

def k_max_neighbors(d, edge_index, n_neighbor=1, remove_self=False, return_steps=False, undirected=False):
    d = d.clone()
    fill_value = d.min()-1
    self_edges = edge_index[0] == edge_index[1]
    d[self_edges] = fill_value

    used_edges = torch.zeros_like(self_edges) == 1
    steps = []

    for n in range(n_neighbor):
        edge_n = scatter_max(d, edge_index[0], dim=0)[1]
        edge_n = edge_n[~used_edges[edge_n]]
        used_edges[edge_n] = True
        steps.append( used_edges.clone() )
        d[edge_n] = fill_value

    if return_steps:
        if not remove_self:
            steps = [ step | self_edges for step in steps ]
        if undirected:
            steps = [ mask_undirected(edge_index, step) for step in steps ]
        return steps

    if not remove_self: 
        used_edges = used_edges | self_edges
    if undirected:
        used_edges = mask_undirected(edge_index, used_edges)
    return used_edges

def sample_pair(data, pair):
    data, pair = data.to('cpu'), pair.to('cpu')
    pair = pair.reshape(2,-1)
    labels = torch.zeros(data.num_nodes, 1)
    labels[pair] = 1
    if hasattr(data,'labeled'): data.x = data.x[:,:-1]
    data.x = torch.cat([data.x, labels], dim=-1)
    data.labeled = torch.LongTensor([1])
    data.pair_index = pair

    pair_id = data.node_id[data.pair_index]
    higgs_id = (pair_id+1)//2
    higgs_mask = (higgs_id[1] == higgs_id[0]) & (higgs_id[0] > 0)
    data.pair_y = 1*higgs_mask
    return data.to(config.device)

def sample_cluster(data, cluster):
    data, cluster = data.to('cpu'), cluster.to('cpu')
    cluster = cluster.reshape(4,-1)
    labels = torch.zeros(data.num_nodes, 1)
    labels[cluster] = 1
    if hasattr(data,'labeled'): data.x = data.x[:,:-1]
    data.x = torch.cat([data.x, labels], dim=-1)
    data.labeled = torch.LongTensor([1])
    data.cluster_index = cluster

    cluster_id = data.node_id[data.cluster_index]
    y_id = (cluster_id+3)//4
    y_mask = (y_id[1] == y_id[0]) & (y_id[0] > 0)
    data.cluster_y = 1*y_mask
    return data.to(config.device)