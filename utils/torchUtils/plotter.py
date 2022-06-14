import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx

from ..selectUtils import *
from ..utils import *
from ..plotUtils import *
from .gnn import graph_pred, mask_graph_edges

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
                  for n in g.nodes] if node_attr is not None else 'tab:grey'
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
    node_truth = nx.get_node_attributes(g,'y')
    node_color = {n: 'tab:blue' if node_truth[n] else 'tab:grey'  for n in g.nodes}
    pair_score = {n: 0 for n in g.nodes}
    
    edge_truth = nx.get_edge_attributes(g,'edge_y')
    edge_color = {e: 'tab:blue' if edge_truth[e] else 'black' for e in g.edges}
    width = {e: 0 for e in g.edges}
    
    true_var = 'edge_y'
    pred_var = 'edge_pred' if 'edge_pred' in g.get_edge_data(0,1) else 'edge_y'
    
    scores = {(ni, nj): g.get_edge_data(ni, nj)[pred_var] for (ni, nj) in g.edges}
    sorted_scores = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    for (ni, nj) in map(lambda kv: kv[0], sorted_scores):
        paired = g.get_edge_data(ni, nj)[pred_var]
        
        width[(ni,nj)] = paired

        if pair_score[ni] < paired and pair_score[nj] < paired:
            node_pairs[ni] = (ni,nj)
            node_pairs[nj] = (ni,nj)
            
            pair_score[ni] = paired
            pair_score[nj] = paired

    npairs = 0
    for (ni, nj) in map(lambda kv: kv[0], sorted_scores):
        if npairs == 4: break
        if node_pairs[ni] == -1 or node_pairs[nj] == -1:
            continue
        if node_pairs[ni] == node_pairs[nj]:
            npairs += 1
            node_color[ni] = 'tab:orange' if node_truth[ni] else 'tab:red'
            node_color[nj] = 'tab:orange' if node_truth[nj] else 'tab:red'
            
            edge_color[(ni, nj)] = 'tab:orange' if edge_truth[(ni, nj)] else 'tab:red'

    node_color = list(node_color.values())
    edge_color = list(edge_color.values())
    width = np.array(list(width.values()))
    width = width/np.std(width)
    return dict(node_color=node_color, edge_color=edge_color, width=np.array(width))


def display_graph(g, pos='xy', sizing=1, coloring='paired', show_detector=False, figax=None):
    """Create a graph display

    Args:
        g (torch_geometric.data.Data): Pytorch graph data
        pos (str, optional): What 2D coorinates to use for positioning
        sizing (str, int, optional): What node attr to use as sizing, y uses target, int uses x at that index
        coloring (str, optional): type of coloring to use, paired or default. Defaults to 'paired'
        show_detector (bool, optional): If xye position, draws the barrel or endcap of the detector
    """
    if figax is None:
        figax = plt.subplots()
    fig, ax = figax
    plt.sca(ax)

    posmap = dict(
        e=lambda attr: attr[2],
        p=lambda attr: attr[3],
        x=lambda attr: (np.cos(2*np.pi*attr[3])+1)/2,
        y=lambda attr: (np.sin(2*np.pi*attr[3])+1)/2,
    )

    colorings = dict(paired=paired_coloring, default=default_coloring)
    
    edge_attrs = ['edge_attr', 'edge_y']
    if hasattr(g,'edge_pred'): edge_attrs += ['edge_pred']

    g = nx.Graph(to_networkx(g, node_attrs=['x', 'y'], 
                             edge_attrs=edge_attrs, remove_self_loops=True))

    node_pos = np.array([[posmap[p](g.nodes[n]['x'])
                        for p in pos] for n in g.nodes])

    get_size = dict(
        y=lambda node: node['y'],
        x=lambda node, sizing=sizing: node['x'][sizing],
    )

    if type(sizing) is int:
        sizing = 'x'

    node_size = np.array([get_size[sizing](g.nodes[n]) for n in g])
    node_size = node_size/np.std(node_size)
    node_size = 1000*(node_size-np.min(node_size)) / \
        (np.max(node_size)-np.min(node_size))
    node_size = np.where(node_size > 100, node_size, 100)

    coloring = colorings.get(coloring, lambda g: {'node_color':'tab:grey'})(g)

    nx.draw(g,  node_size=node_size, **coloring, alpha=0.8)

    # plt.gca().set(xlim=(-0.2, 1.2), ylim=(-0.2, 1.2))
    plt.gca().set_aspect('equal')

    if show_detector and all(r in 'xye' for r in pos):
        if sorted(pos) == ['x', 'y']:
            plt.gca().add_patch(plt.Circle((0.5, 0.5), 0.5, fill=False))
        else:
            plt.plot([[0], [0]], 'k')
            plt.plot([[1], [1]], 'k')
            plt.plot([[0.5], [0.5]], 'k-.')

    return fig, ax


def display_pred(model, g, *args, **kwargs):
    """Plots graph prediction

    Args:
        model (pyTorch Model): Model to use to predict
        g (Graph Data): pyTorch graph data
    """
    fig, axs = plt.subplots(ncols=2, figsize=(10, 5))
    display_graph(g, *args, **kwargs, figax=(fig, axs[0]))
    axs[0].set(title="True")
    g_pred = graph_pred(model, g)
    display_graph(g_pred, *args, **kwargs, figax=(fig, axs[1]))
    axs[1].set(title="Pred")

def plot_aucroc(roc_metrics,tag="",figax=None):
    if figax is None: figax = plt.subplots()

    fpr, tpr, auc = roc_metrics.get_values()
    graph_arrays([fpr], [tpr], xlabel=f"{tag} False Positive", ylabel=f"{tag} True Positive",
                 title=f"AUC: {auc:.3}", ylim=(-0.05,1.05), figax=figax)
    return figax


def plot_graph_auroc(node_metrics, edge_metrics):
    fig, axs = plt.subplots(ncols=2, figsize=(16, 5))

    plot_aucroc(node_metrics,'Node',figax=(fig,axs[0]))
    plot_aucroc(edge_metrics,'Edge',figax=(fig,axs[1]))

    return fig, axs

def draw_data(data, edge_mask=None, width=None, figax=None, edge_labels=None, undirected=False, **kwargs):
    if figax is None: figax = plt.subplots()
    fig, ax = figax
    plt.sca(ax)

    # if width is not None:
    #     width = gnn.attr_undirected(data.edge_index, width)

    self_edges = data.edge_index[0] == data.edge_index[1]

    if edge_mask is None: edge_mask = ~self_edges
    edge_mask = edge_mask & (~self_edges)
    uptri = data.edge_index[0] < data.edge_index[1]
    
    # edge_mask = mask_undirected(data, edge_mask)
    data = mask_graph_edges(data.clone(), edge_mask)

    if undirected:
        edge_mask = edge_mask & uptri
    if width is not None:
        width = width[edge_mask]
    
    # if width is not None:
    #     width = width[data.edge_index[0]<data.edge_index[1]]

    for key, value in kwargs.items():
        if value.shape[0] == edge_mask.shape[0]:
            kwargs[key] = value[edge_mask].numpy()

    graph = to_networkx(data, remove_self_loops=True, to_undirected=undirected)
    pos = nx.circular_layout(graph)
    nx.draw(graph, pos, node_color=(data.node_id+1)//2, width=width, **kwargs)

    if edge_labels is not None:
        edge_labels = edge_labels[edge_mask]
        edge_labels = dict([( ( int(n1), int(n2)), int(label)) for (n1,n2),label in zip(data.edge_index.T, edge_labels)])
        nx.draw_networkx_edge_labels(graph, pos, edge_labels)

    return fig, ax
