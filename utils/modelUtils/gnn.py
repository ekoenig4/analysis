from itertools import cycle
from typing import Callable, Union

import matplotlib.pyplot as plt
import networkx as nx
import torch
import torch_geometric
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.typing import Adj, PairTensor
from torch_geometric.utils import to_networkx

from ..classUtils.GraphDataset import Dataset
from ..selectUtils import *
from ..utils import *


def to_tensor(tensor, gpu=False):
    if not torch.is_tensor(tensor):
        tensor = torch.Tensor(tensor)
    if gpu:
        return tensor.to('cuda:0')
    return tensor


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

# Default Coloring


def default_coloring(g, node_attr=None):
    """Default coloring which colors nodes based on a node attr

    Args:
        g (torch_geometric.data.Data): Pytroch garph data
        node_attr (int, optional): Index of node attr to color nodes with. Defaults to None.

    Returns:
        dict: Dictionary attrs for coloring in networkx
    """
    node_color = [g.nodes[n]['x'][node_attr]
                  for n in g.nodes] if node_attr is not None else None
    width = [g.get_edge_data(ni, nj)['edge_y'] for ni, nj in g.edges]
    return dict(node_color=node_color, width=np.array(width))


# Paired Coloring


def paired_coloring(g):
    """Paired coloring which colors the highest scoring node pairs

    Args:
        g (torch_geometric.data.Data): Pytorch graph data

    Returns:
        dict: Dictionary attrs for coloring in networkx
    """
    node_pairs = {n: -1 for n in g.nodes}
    node_color = {n: 0 for n in g.nodes}
    node_score = {n: 0 for n in g.nodes}
    edge_color = {e: 0 for e, _ in enumerate(g.edges)}
    width = {e: 0 for e, _ in enumerate(g.edges)}
    color = 0

    scores = {(e, (ni, nj)): g.get_edge_data(ni, nj)[
        'edge_y'] for e, (ni, nj) in enumerate(g.edges)}
    sorted_scores = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)

    for e, (ni, nj) in map(lambda kv: kv[0], sorted_scores):
        paired = g.get_edge_data(ni, nj)['edge_y']
        width[e] = paired

        if node_score[ni] < paired and node_score[nj] < paired:
            node_pairs[ni] = e
            node_pairs[nj] = e
            node_score[ni] = paired
            node_score[nj] = paired

    for e, (ni, nj) in enumerate(g.edges):
        if node_pairs[ni] == -1 or node_pairs[nj] == -1:
            continue
        if node_pairs[ni] == node_pairs[nj]:
            color += 1
            node_color[ni] = color
            node_color[nj] = color
            edge_color[e] = color
        if color >= 3:
            break

    node_color = list(node_color.values())
    edge_color = list(edge_color.values())
    width = list(width.values())
    return dict(node_color=node_color, edge_color=edge_color, width=np.array(width))


def display_graph(g, pos='xy', coloring='paired', show_detector=False):
    """Create a graph display

    Args:
        g (torch_geometric.data.Data): Pytorch graph data
        pos (str, optional): What 2D coorinates to use for positioning
        coloring (str, optional): type of coloring to use, paired or default. Defaults to 'paired'
        show_detector (bool, optional): If xzy position, draws the barrel or endcap of the detector
    """

    posmap = dict(
        e=lambda attr: attr[2],
        p=lambda attr: attr[3],
        x=lambda attr: attr[5],
        y=lambda attr: attr[6],
        z=lambda attr: attr[7],
        r=lambda attr: np.sqrt(attr[5]**2 + attr[6]**2)
    )

    colorings = dict(paired=paired_coloring, default=default_coloring)

    g = to_networkx(g, node_attrs=['x', 'y'], edge_attrs=[
                    'edge_attr', 'edge_y'], to_undirected=True, remove_self_loops=True)

    node_pos = np.array([[posmap[p](g.nodes[n]['x'])
                        for p in pos] for n in g.nodes])
    node_size = np.array([g.nodes[n]['x'][1] for n in g])
    node_size /= np.std(node_size)
    coloring = colorings[coloring](g)

    nx.draw(g, node_pos, node_size=1000*node_size, **coloring, alpha=0.8)

    plt.gca().set(xlim=(-0.2, 1.2), ylim=(-0.2, 1.2))
    plt.gca().set_aspect('equal')

    if show_detector and all(r in 'xyz' for r in pos):
        if sorted(pos) == ['x', 'y']:
            plt.gca().add_patch(plt.Circle((0.5, 0.5), 0.5, fill=False))
        else:
            plt.plot([[0], [0]], 'k')
            plt.plot([[1], [1]], 'k')
            plt.plot([[0.5], [0.5]], 'k-.')


def graph_pred(model, g, *args, **kwargs):
    edge_pred = model(g)
    if type(edge_pred) is tuple:
        _,edge_pred = edge_pred 

    edge_pred = (1-edge_pred/edge_pred.sum(dim=1, keepdim=True))[:, 1]
    g_pred = Data(x=g.x, edge_index=g.edge_index,
                  edge_attr=g.edge_attr, y=g.y, edge_y=edge_pred)

    fig, axs = plt.subplots(ncols=2, figsize=(10, 5))
    plt.sca(axs[0])
    axs[0].set(title="True")
    display_graph(g, *args, **kwargs)
    plt.sca(axs[1])
    axs[1].set(title="Pred")
    display_graph(g_pred, *args, **kwargs)
