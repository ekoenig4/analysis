from torch_geometric.typing import PairTensor, Adj
from typing import Union, Callable
import torch_geometric
from torch import Tensor
import matplotlib.pyplot as plt
import networkx as nx
import torch
from ..utils import *
from ..selectUtils import *
from ..classUtils.GraphDataset import Dataset




class EdgeConv(torch_geometric.nn.EdgeConv):
    aggr_funcs = dict(
        max=lambda tensor: tensor.max(dim=-1)[0],
        min=lambda tensor: tensor.min(dim=-1)[0],
        mean=lambda tensor: tensor.mean(dim=-1),
    )

    def __init__(self, nn: Callable, aggr: str = 'max', edge_aggr: str = 'max', return_with_edges: bool = False, return_only_edges: bool = False, **kwargs):
        super(EdgeConv, self).__init__(nn, aggr, **kwargs)
        self.edge_x: Tensor = Tensor

        assert edge_aggr in ['max', 'mean', 'min', None]
        self.edge_aggr = edge_aggr

        self.return_with_edges = return_with_edges
        self.return_only_edges = return_only_edges

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj, edge_x: Tensor) -> Tensor:
        if isinstance(x, Tensor):
            x: PairTensor = (x, x)
        # propagate_type: (x: PairTensor)
        x = self.propagate(edge_index, x=x, edge_x=edge_x)
        if self.return_with_edges or self.return_only_edges:
            if self.edge_aggr is None:
                edge_x = torch.cat(
                    [x[edge_index[1]], x[edge_index[0]], self.edge_x], dim=-1)
            else:
                edge_x = torch.cat(
                    [x[edge_index[1]][:, :, None], x[edge_index[0]][:, :, None], self.edge_x[:, :, None]], dim=-1)
                edge_x = self.aggr_funcs[self.edge_aggr](edge_x)

            if self.return_with_edges:
                return x, edge_x
            if self.return_only_edges:
                return edge_x

        return x

    def message(self, x_i: Tensor, x_j: Tensor, edge_x: Tensor) -> Tensor:
        self.edge_x = self.nn(torch.cat([x_i, x_j - x_i, edge_x], dim=-1))
        return self.edge_x


def plot_jetgraph(dr, labels):
    if not type(dr) is np.ndarray:
        dr = dr.to_numpy()
    if not type(labels) is np.ndarray:
        labels = labels.to_numpy()
    colormap = {-1: 'grey', 0: 'lightblue', 1: 'lightblue',
                2: 'pink', 3: 'pink', 4: 'lightgreen', 5: 'lightgreen'}
    node_color = [colormap[label] for label in labels]
    G = nx.from_numpy_matrix(dr)

    def add_length(u, v):
        length = 1./G[u][v]['weight']
        G[u][v]['length'] = length
        return length
    length = [add_length(u, v) for u, v in G.edges()]
    G = nx.relabel_nodes(G, {i: j for i, j in enumerate(labels)}, copy=True)
    nx.draw(G, with_labels=True, font_weight='bold', pos=nx.spring_layout(
        G, weight='length'), width=length, node_color=node_color)
    plt.show()



