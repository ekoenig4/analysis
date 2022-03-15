from torch_geometric.nn import Linear
import torch
import torch.nn.functional as F

from .cpp_geometric import layers
from .LightningModel import LightningModel


class GCN(LightningModel):
    def __init__(self, dataset, nn1_out=32, nn2_out=128, **kwargs):
        super().__init__(dataset, **kwargs)

        nn1 = torch.nn.Sequential(
            Linear(2*len(dataset.node_attr_names) +
                   len(dataset.edge_attr_names), nn1_out),
            torch.nn.ELU()
        )

        self.conv1 = layers.EdgeConv(
            nn1, edge_aggr=None, return_with_edges=True)

        nn2 = torch.nn.Sequential(
            Linear(5*nn1_out, nn2_out),
            torch.nn.ELU()
        )

        self.conv2 = layers.EdgeConv(
            nn2, edge_aggr=None, return_with_edges=True)

        self.edge_seq = torch.nn.Sequential(
            Linear(3*nn2_out, 2),
        )

        self.node_seq = torch.nn.Sequential(
            Linear(nn2_out, 2),
        )

    def forward(self, data):
        if type(data) is list:
            data = data[0]
        if type(data) is not tuple:
            data = (data.x, data.edge_index, data.edge_attr)
        x, edge_index, edge_attr = data

        x, edge_attr = self.conv1(x, edge_index, edge_attr)
        x, edge_attr = self.conv2(x, edge_index, edge_attr)
        x, edge_attr = self.node_seq(x), self.edge_seq(edge_attr)

        return F.log_softmax(x, dim=1), F.log_softmax(edge_attr, dim=1)


class GoldenGCN(LightningModel):
    def __init__(self, dataset, nn1_out=32, nn2_out=128, **kwargs):
        super().__init__(dataset, **kwargs)
        self.save_hyperparameters('nn1_out', 'nn2_out')

        n_in_node = len(dataset.node_attr_names)
        n_in_edge = len(dataset.edge_attr_names)
        
        self.conv1 = layers.GCNConvMSG(n_in_node=n_in_node, n_in_edge=n_in_edge, n_out=nn1_out)
        self.relu1 = layers.GCNRelu()
        self.conv2 = layers.GCNConvMSG(
            n_in_node=nn1_out, n_in_edge=3*nn1_out, n_out=nn2_out)
        self.relu2 = layers.GCNRelu()

        self.node_linear = layers.NodeLinear(nn2_out, 2)
        self.edge_linear = layers.EdgeLinear(3*nn2_out, 2)
        self.log_softmax = layers.GCNLogSoftmax()

    def forward(self, data):
        if type(data) is list:
            data = data[0]
        if type(data) is not tuple:
            data = (data.x, data.edge_index, data.edge_attr)
        x, edge_index, edge_attr = data

        x, edge_attr = self.conv1(x, edge_index, edge_attr)
        x, edge_attr = self.relu1(x, edge_index, edge_attr)

        x, edge_attr = self.conv2(x, edge_index, edge_attr)
        x, edge_attr = self.relu2(x, edge_index, edge_attr)

        x, edge_attr = self.node_linear(x, edge_index, edge_attr)
        x, edge_attr = self.edge_linear(x, edge_index, edge_attr)

        x, edge_attr = self.log_softmax(x, edge_index, edge_attr)

        return x, edge_attr
