import numba as nb
import awkward as ak
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import dense_to_sparse
import numpy as np

from ..selectUtils import *
from ..utils import get_collection


class ScaleNodeAttrs:
    def fit(self, node_attrs):
        self.nfeatures = node_attrs[0].to_numpy().shape[-1]
        self.minims = ak.Array([ak.min(node_attrs[:, :, i])
                               for i in range(self.nfeatures)])[None, None]
        self.maxims = ak.Array([ak.max(node_attrs[:, :, i])
                               for i in range(self.nfeatures)])[None, None]
        return self

    def transform(self, node_attrs):
        return (node_attrs - self.minims)/(self.maxims-self.minims)

    def inverse(self, node_attrs):
        return (self.maxims-self.minims)*node_attrs + self.minims


class ScaleEdgeAttrs:
    def fit(self, x):
        self.nfeatures = x[0].to_numpy().shape[-1]
        self.minims = ak.Array([ak.min(x[:, :, :, i])
                               for i in range(self.nfeatures)])[None, None, None]
        self.maxims = ak.Array([ak.max(x[:, :, :, i])
                               for i in range(self.nfeatures)])[None, None, None]
        return self

    def transform(self, x):
        return (x - self.minims)/(self.maxims-self.minims)

    def inverse(self, x):
        return (self.maxims-self.minims)*x + self.minims


def get_node_attrs(jets, attrs=["m", "pt", "eta", "phi", "btag"]):
    features = ak.concatenate([attr[:, :, None]
                              for attr in ak.unzip(jets[attrs])], axis=-1)
    return features


def get_node_targs(jets):
    targets = 1*(jets.signalId > -1)
    return targets


def get_edge_attrs(jets, attrs=["dr", "dx"]):
    dr = calc_dr(jets.eta[:, :, None], jets.phi[:, :, None],
                 jets.eta[:, None], jets.phi[:, None])[:, :, :, None]
    dx = np.sqrt((jets.x[:, None] - jets.x[:, :, None])**2 + (jets.y[:, None] -
                 jets.y[:, :, None])**2 + (jets.z[:, None] - jets.z[:, :, None])**2)[:, :, :, None]
    return ak.concatenate([dr, dx], axis=-1)


def get_edge_targs(jets):
    diff = np.abs(jets.signalId[:, None] - jets.signalId[:, :, None])
    add = jets.signalId[:, None] + jets.signalId[:, :, None]
    mod2 = add % 2

    paired = (diff*mod2 == 1) & ((add == 1) | (add == 5) | (add == 9))
    return 1*paired


def build_features(tree, node_attr_names=["m", "pt", "eta", "phi", "btag", "x", "y", "z"], edge_attr_names=["dr"]):
    tree.extend(
        jet_x=np.cos(tree.jet_phi),
        jet_y=np.sin(tree.jet_phi),
        jet_z=1/np.tan(2*np.arctan(np.exp(-tree.jet_eta)))
    )

    jets = get_collection(tree, 'jet', named=False)
    node_attrs = get_node_attrs(jets, node_attr_names)
    node_targs = get_node_targs(jets)
    edge_attrs = get_edge_attrs(jets, edge_attr_names)
    edge_targs = get_edge_targs(jets)

    node_scaler = ScaleNodeAttrs().fit(node_attrs)
    node_attrs = node_scaler.transform(node_attrs)

    edge_scaler = ScaleEdgeAttrs().fit(edge_attrs)
    edge_attrs = edge_scaler.transform(edge_attrs)

    node_class_weights = np.array(
        [np.sum(node_targs == 0), np.sum(node_targs == 1)])
    node_class_weights = np.max(node_class_weights)/node_class_weights

    edge_class_weights = np.array(
        [np.sum(edge_targs == 0), np.sum(edge_targs == 1)])
    edge_class_weights = np.max(edge_class_weights)/edge_class_weights

    node_features = ak.concatenate(
        [node_attrs, node_targs[:, :, None]], axis=-1)
    edge_features = ak.concatenate(
        [edge_attrs, edge_targs[:, :, :, None]], axis=-1)

    return (node_features, node_scaler), (edge_features, edge_scaler), (node_class_weights, edge_class_weights), (node_attr_names, edge_attr_names)


def build_graph(node_features, edge_features):
    node_attrs = torch.Tensor(node_features[:, :-1])
    node_targs = torch.Tensor(node_features[:, -1]).long().reshape(-1)
    n_nodes = len(node_targs)

    edge_attrs = torch.Tensor(
        edge_features[:, :, :-1]).reshape(n_nodes*n_nodes, -1)
    edge_targs = torch.Tensor(edge_features[:, :, -1]).long().reshape(-1)
    edge_index, _ = dense_to_sparse(torch.full((n_nodes, n_nodes), 1))

    return Data(x=node_attrs, y=node_targs, edge_index=edge_index, edge_attr=edge_attrs, edge_y=edge_targs)


@nb.jit(forceobj=True)
def build_dataset(node_features, edge_features):
    return [build_graph(node_attrs, edge_attrs) for node_attrs, edge_attrs in zip(node_features, edge_features)]


class Dataset(InMemoryDataset):
    def __init__(self, root, tree=None, transform=None, pre_transform=None):
        self.processed_tree = tree

        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.node_attr_names, self.edge_attr_names = torch.load(
            self.processed_paths[1])
        self.node_class_weights, self.edge_class_weights = torch.load(
            self.processed_paths[2])
        self.filelist = torch.load(self.processed_paths[3])
        self.node_scaler, self.edge_scaler = torch.load(
            self.processed_paths[4])

    @property
    def processed_file_names(self):
        return ['graphs.pt', 'attr_names.pt', 'class_weights.pt', 'filelist.pt', 'scalers.pt']

    def process(self):
        # Read data into huge `Data` list.
        filelist = list(map(lambda f: f.fname, self.processed_tree.filelist))
        (node_features, node_scaler), (edge_features, edge_scaler), (node_class_weights, edge_class_weights), (node_attr_names, edge_attr_names) = build_features(
            self.processed_tree)

        data_list = build_dataset(node_features, edge_features)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)

        torch.save((data, slices), self.processed_paths[0])
        torch.save((node_attr_names, edge_attr_names), self.processed_paths[1])
        torch.save((node_class_weights, edge_class_weights),
                   self.processed_paths[2])
        torch.save(filelist, self.processed_paths[3])
        torch.save((node_scaler, edge_scaler), self.processed_paths[4])
