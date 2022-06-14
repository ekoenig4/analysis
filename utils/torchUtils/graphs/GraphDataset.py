
import awkward as ak
import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch.utils.data import ConcatDataset

from ...utils import get_collection
from ..torchscript import build_dataset
from ..gnn import *
from .GraphTransforms import *


class ScaleAttrs:
    def __init__(self):
        self.functions = dict(
            normalize=self.normalize,
            standardize=self.standardize
        )
        self.inv_functions = dict(
            normalize=self.inv_normalize,
            standardize=self.inv_standardize,
        )
        
    def fit(self, x):
        self.nfeatures = x[0].to_numpy().shape[-1]
        self.minims = np.array([ak.min(x[:, :, i])
                               for i in range(self.nfeatures)])
        self.maxims = np.array([ak.max(x[:, :, i])
                               for i in range(self.nfeatures)])
        self.means = np.array([ak.mean(x[:, :, i])
                               for i in range(self.nfeatures)])
        self.stds = np.array([ak.std(x[:, :, i])
                               for i in range(self.nfeatures)])
        
        return self
    
    def to_torch(self):
        self.minims = torch.Tensor(self.minims).float()
        self.maxims = torch.Tensor(self.maxims).float()
        self.means = torch.Tensor(self.means).float()
        self.stds = torch.Tensor(self.stds).float()
        
    def to_numpy(self): pass
        
    
    def _process_awk(self,x, operation):
        n_nodes = ak.num(x, axis=1)
        x = ak.flatten(x, axis=1).to_numpy()
        x = operation(x)
        return ak.unflatten(x, n_nodes)
    
    def _process_torch(self, x, operation):
        return operation(x).float()

    def _process(self, x, operation):
        if isinstance(x, ak.Array): return self._process_awk(x, operation)
        if isinstance(x, torch.Tensor): return self._process_torch(x, operation)
        return operation(x)
        
    def normalize(self, x):
        operation = lambda x : (x - self.minims)/(self.maxims - self.minims)
        return self._process(x, operation)
    
    def standardize(self, x):
        operation = lambda x : (x - self.means)/self.stds
        return self._process(x, operation)
    
    def inv_normalize(self,x):
        operation = lambda x : (self.maxims-self.minims)*x + self.minims
        return self._process(x, operation)

    def inv_standardize(self,x):
        operation = lambda x : self.stds*x + self.means
        return self._process(x, operation)

    def transform(self, x, type='normalize'):
        return self.functions[type](x)        
        
    def inverse(self, x, type='normalize'):
        return self.inv_functions[type](x)        
        

def graph_to_torch(graph):
    return Data(x=graph.x, y=graph.y, edge_index=graph.edge_index, edge_attr=graph.edge_attr, edge_y=graph.edge_y, node_id=graph.node_id, edge_id=graph.edge_id)


def prepare_features(attrs, targs):
    slices = torch.from_numpy(ak.num(attrs, axis=1).to_numpy())
    slices = torch.cat([torch.Tensor([0]), slices.cumsum(dim=0)]).long()
    attrs = torch.from_numpy(ak.flatten(attrs, axis=1).to_numpy()).float()
    targs = torch.from_numpy(ak.flatten(targs, axis=1).to_numpy()).long()
    return attrs, targs, slices

def clean_features(features, fill_value=0):
    is_inf = np.abs(features) == np.inf
    is_nan = np.isnan(features)

    features = ak.where( is_inf, fill_value, features)
    features = ak.where( is_nan, fill_value, features)
    return features


def get_node_attrs(jets, attrs=["mRegressed", "ptRegressed", "eta", "phi", "btag"]):
    features = ak.concatenate([attr[:, :, None]
                              for attr in ak.unzip(jets[attrs])], axis=-1)
    return clean_features(features)


def get_node_targs(jets):
    targets = jets.signalId + 1
    return targets


def get_edge_attrs(jets, attrs=["m","pt","eta","phi","dr"]):
    deta = ak.flatten(calc_deta(jets.eta[:,:,None],jets.eta[:,None]),axis=2)
    dphi = ak.flatten(calc_dphi(jets.phi[:,:,None],jets.phi[:,None]),axis=2)
    dr = np.sqrt(deta**2 + dphi**2)

    p4_1 = vector.obj(pt=jets.ptRegressed[:,:,None],eta=jets.eta[:,:,None],phi=jets.phi[:,:,None],m=jets.mRegressed[:,:,None])
    p4_2 = vector.obj(pt=jets.ptRegressed[:,None],eta=jets.eta[:,None],phi=jets.phi[:,None],m=jets.mRegressed[:,None])
    res = p4_1 + p4_2

    m = ak.flatten(res.m,axis=2)
    pt = ak.flatten(res.pt,axis=2)
    eta = ak.flatten(res.eta,axis=2)
    phi = ak.flatten(res.phi,axis=2)
    
    edges = ak.zip(dict(m=m,pt=pt,eta=eta,phi=phi,dr=dr),depth_limit=1)
    
    features = ak.concatenate([attr[:, :, None]
                              for attr in ak.unzip(edges[attrs])], axis=-1)

    return clean_features(features)


def get_edge_targs(jets):
    diff = np.abs(jets.signalId[:, None] - jets.signalId[:, :, None])
    add = jets.signalId[:, None] + jets.signalId[:, :, None]
    mod2 = add % 2

    paired = (diff*mod2 == 1) & ((add == 1) | (add == 5) | (add == 9) | (add == 13))

    ids = 1*(add == 1) + 2*(add == 5) + 3*(add == 9) + 4*(add == 13)
    target = ak.where(paired, ids, 0)
    return ak.flatten(target, axis=2)


def build_node_features(jets, node_attr_names=["mRegressed", "ptRegressed", "eta", "phi", "btag"]):
    node_attrs = get_node_attrs(jets, node_attr_names)
    node_targs = get_node_targs(jets)
    return node_attrs, node_targs, node_attr_names


def build_edge_features(jets, edge_attr_names=["m","pt","eta","phi","dr"]):
    edge_attrs = get_edge_attrs(jets, edge_attr_names)
    edge_targs = get_edge_targs(jets)
    return edge_attrs, edge_targs, edge_attr_names


def build_features(tree, node_attr_names=["mRegressed", "ptRegressed", "eta", "phi", "btag"], edge_attr_names=["m", "pt", "eta", "phi", "dr"]):
    jets = get_collection(tree, 'jet', False)
    return build_node_features(jets, node_attr_names), build_edge_features(jets, edge_attr_names)


def scale_attrs(attrs, scaler=None):
    if scaler is None:
        scaler = ScaleAttrs().fit(attrs)
    attrs = scaler.transform(attrs)
    return attrs, scaler


def get_class_weights(node_targs, edge_targs):
    pos_node_targs = np.sum(node_targs == 1)
    neg_node_targs = np.sum(node_targs == 0)
    num_nodes = pos_node_targs + neg_node_targs
    node_class_weights = max(neg_node_targs, pos_node_targs) / \
        np.array([neg_node_targs, pos_node_targs])

    pos_edge_targs = np.sum(edge_targs == 1)/2
    neg_edge_targs = (np.sum(edge_targs == 0)-num_nodes)/2
    num_edges = pos_edge_targs + neg_edge_targs
    edge_class_weights = max(neg_edge_targs, pos_edge_targs) / \
        np.array([neg_edge_targs, pos_edge_targs])

    type_class_weights = max(num_nodes, num_edges) / \
        np.array([num_nodes, num_edges])

    return node_class_weights, edge_class_weights, type_class_weights

def concat_dataset(array,**dataset_kwargs):
    if type(array[0]) == str:
        array = [Dataset(fn,**dataset_kwargs) for fn in array]
    return ConcatDataset(array)

def _insert_transform(self, transform):
    if self.transform is None:
        self.transform = Transform(transform)
    elif isinstance(self.transform, Transform):
        self.transform.insert(transform)
    else:
        self.transform = Transform(transform, self.transform)
        
def _append_transform(self, transform):
    if self.transform is None:
        self.transform = Transform(transform)
    elif isinstance(self.transform, Transform):
        self.transform.append(transform)
    else:
        self.transform = Transform(transform, self.transform)
    

class Dataset(InMemoryDataset):
    def __init__(self, root, tree=None, template=None, transform=None, scale='standardize', node_mask=[], edge_mask=[], make_template=False):
        self.tree = tree
        self.make_template = make_template
        self.scale = scale
        self.node_scaler = template.node_scaler if template is not None else None
        self.edge_scaler = template.edge_scaler if template is not None else None
        self.node_class_weights = template.node_class_weights if template is not None else None
        self.edge_class_weights = template.edge_class_weights if template is not None else None
        self.type_class_weights = template.type_class_weights if template is not None else None
        self.node_attr_names = template.node_attr_names if template is not None else None
        self.edge_attr_names = template.edge_attr_names if template is not None else None

        super().__init__(root, transform)
        
        self.node_attr_names, self.edge_attr_names = torch.load(
            self.processed_paths[0])
        self.node_class_weights, self.edge_class_weights, self.type_class_weights = torch.load(
            self.processed_paths[1])
        self.filelist = torch.load(self.processed_paths[2])
        self.node_scaler, self.edge_scaler = torch.load(
            self.processed_paths[3])
        
        if make_template:
            if scale not in (None,'raw'):
                scale = scale_graph(self.node_scaler, self.edge_scaler, scale)
                _insert_transform(self, scale)
                    
            if any(node_mask+edge_mask):
                mask = mask_graph(self.node_attr_names, node_mask, self.edge_attr_names, edge_mask)
                self.node_attr_names = mask.node_features
                self.edge_attr_names = mask.edge_features
                _append_transform(self,mask)
        
        else:
            self.data, self.slices = torch.load(self.processed_paths[4])

    @property
    def processed_file_names(self):
        if self.make_template:
            return ['attr_names.pt', 'class_weights.pt', 'filelist.pt', 'scalers.pt']
        return ['attr_names.pt', 'class_weights.pt', 'filelist.pt', 'scalers.pt','graphs.pt']

    def process(self):
        if self.make_template:
            print("Building Template Dataset...")
        # Read data into huge `Data` list.
        filelist = list(map(lambda f: f.fname, self.tree.filelist))

        print("Building Features...")
        jets = get_collection(self.tree, 'jet', False)

        node_kwargs = {}
        if self.node_attr_names is not None:
            node_kwargs['node_attr_names'] = self.node_attr_names
        edge_kwargs = {}
        if self.edge_attr_names is not None:
            edge_kwargs['edge_attr_names'] = self.edge_attr_names

        node_attrs, node_targs, node_attr_names = build_node_features(
            jets, **node_kwargs)
        edge_attrs, edge_targs, edge_attr_names = build_edge_features(
            jets, **edge_kwargs)
        _, node_scaler = scale_attrs(node_attrs, self.node_scaler)
        _, edge_scaler = scale_attrs(edge_attrs, self.edge_scaler)

        if self.node_class_weights is None or self.edge_class_weights is None or self.type_class_weights is None:
            node_class_weights, edge_class_weights, type_class_weights = get_class_weights(
                node_targs, edge_targs)
        else:
            node_class_weights, edge_class_weights, type_class_weights = self.node_class_weights, self.edge_class_weights, self.type_class_weights

        node_attrs, node_targs, node_slices = prepare_features(
            node_attrs, node_targs)
        edge_attrs, edge_targs, edge_slices = prepare_features(
            edge_attrs, edge_targs)

        assert node_attrs.type(
        ) == 'torch.FloatTensor', f"Expected node_attrs of type torch.FloatTensor, but got {node_attrs.type()}"
        assert node_targs.type(
        ) == 'torch.LongTensor', f"Expected node_targs of type torch.LongTensor, but got {node_targs.type()}"
        assert node_slices.type(
        ) == 'torch.LongTensor', f"Expected node_slices of type torch.LongTensor, but got {node_slices.type()}"

        assert edge_attrs.type(
        ) == 'torch.FloatTensor', f"Expected edge_attrs of type torch.FloatTensor, but got {edge_attrs.type()}"
        assert edge_targs.type(
        ) == 'torch.LongTensor', f"Expected edge_targs of type torch.LongTensor, but got {edge_targs.type()}"
        assert edge_slices.type(
        ) == 'torch.LongTensor', f"Expected edge_slices of type torch.LongTensor, but got {edge_slices.type()}"

        if not self.make_template:
            print("Building Dataset...")
            data_list = build_dataset(
                node_attrs, node_targs, node_slices, edge_attrs, edge_targs, edge_slices)

            data_list = [graph_to_torch(graph) for graph in data_list]

            data, slices = self.collate(data_list)

        print("Saving Dataset...")
        torch.save((node_attr_names, edge_attr_names), self.processed_paths[0])
        torch.save((node_class_weights, edge_class_weights, type_class_weights),
                   self.processed_paths[1])
        torch.save(filelist, self.processed_paths[2])
        torch.save((node_scaler, edge_scaler), self.processed_paths[3])
        if not self.make_template:
            torch.save((data, slices), self.processed_paths[4])

    def build_graphs(self, tree):
        (node_attrs, node_targs, node_attr_names), (edge_attrs,
                                                    edge_targs, edge_attr_names) = build_features(tree, self.node_attr_names, self.edge_attr_names)
        # node_attrs = self.node_scaler.transform(node_attrs)
        # edge_attrs = self.edge_scaler.transform(edge_attrs)

        node_attrs, node_targs, node_slices = prepare_features(
            node_attrs, node_targs)
        edge_attrs, edge_targs, edge_slices = prepare_features(
            edge_attrs, edge_targs)
        data_list = build_dataset(
            node_attrs, node_targs, node_slices, edge_attrs, edge_targs, edge_slices)

        if self.transform is not None:
            return [self.transform(graph_to_torch(graph)) for graph in data_list]
        return [graph_to_torch(graph) for graph in data_list]
