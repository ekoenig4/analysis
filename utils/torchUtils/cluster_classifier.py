import torch
from torch import Tensor
from torch.nn import Module
import torch.nn.functional as F
from torch_geometric.data import Data
import torchmetrics.functional as f_metrics

from . import edge_losses, node_losses, cluster_losses
from .cpp_geometric import *
from .LightningModel import LightningModel

__all__ = [ 
    "GoldenCluster"
]

class ClusterClassifier(LightningModel):
    def __init__(self, loss='std_loss', **kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters('loss')
        self.node_loss = getattr(node_losses, loss)
        self.edge_loss = getattr(edge_losses, loss)
        self.cluster_loss = getattr(cluster_losses,loss)

    def shared_step(self, batch, batch_idx, tag):
        node_o, edge_o, cluster_o = self(batch)
        loss = self.node_loss(self, node_o, batch) + self.edge_loss(self, edge_o, batch) + self.cluster_loss(self, cluster_o, batch)

        node_score = torch.exp(node_o[:,1])
        node_auroc = f_metrics.auroc(node_score, batch.y)

        edge_score = torch.exp(edge_o[:,1])
        edge_auroc = f_metrics.auroc(edge_score, batch.edge_y)

        y1_score = torch.exp(cluster_o[:,1])
        y1_auroc = f_metrics.auroc(y1_score, 1*(batch.cluster_y == 1))

        y2_score = torch.exp(cluster_o[:,2])
        y2_auroc = f_metrics.auroc(y2_score, 1*(batch.cluster_y == 2))

        metrics = dict(
            loss=loss, 
            node_auroc=node_auroc,
            edge_auroc=edge_auroc,
            y1_auroc  =y1_auroc,
            y2_auroc  =y2_auroc
        )
        self.log_scalar(metrics, tag)

        true_node_score = node_score[batch.y == 1]
        fake_node_score = node_score[batch.y == 0]

        true_edge_score = edge_score[batch.edge_y == 1]
        fake_edge_score = edge_score[batch.edge_y == 0]

        true_y1_score = y1_score[batch.cluster_y == 1]
        fake_y1_score = y1_score[batch.cluster_y != 1]

        true_y2_score = y2_score[batch.cluster_y == 2]
        fake_y2_score = y2_score[batch.cluster_y != 2]

        histos = dict(
            true_node_score=true_node_score, fake_node_score=fake_node_score,
            true_edge_score=true_edge_score, fake_edge_score=fake_edge_score,
            true_y1_score  =true_y1_score  , fake_y1_score  =fake_y1_score,
            true_y2_score  =true_y2_score  , fake_y2_score  =fake_y2_score
        )
        self.log_histos(histos, tag)

        return metrics


class GoldenCluster(ClusterClassifier):
    name = 'golden_cluster'
    def __init__(self, nn_conv1_out=64, nn_conv2_out=256, nn_linear_out=128 , **kwargs):
        super().__init__(**kwargs)
        self.conv1 = layers.GCNConvMSG(n_in_node=self.n_in_node, n_in_edge=self.n_in_edge, n_out=nn_conv1_out)
        self.relu1 = layers.GCNRelu()
        self.conv2 = layers.GCNConvMSG(n_in_node=nn_conv1_out, n_in_edge=nn_conv1_out, n_out=nn_conv2_out)
        self.relu2 = layers.GCNRelu()

        self.linear1 = layers.GCNLinear(nn_conv2_out, nn_conv2_out, nn_linear_out)
        self.relu3 = layers.GCNRelu()

        self.linear_o = layers.GCNLinear(nn_linear_out, nn_linear_out, 2)
        self.cluster_linear = torch.nn.Linear(nn_linear_out, 3)

    def forward(self, data : Data) -> Tensor:
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x, edge_attr = self.conv1(x, edge_index, edge_attr)
        x, edge_attr = self.relu1(x, edge_index, edge_attr)

        x, edge_attr = self.conv2(x, edge_index, edge_attr)
        x, edge_attr = self.relu2(x, edge_index, edge_attr)

        x, edge_attr = self.linear1(x, edge_index, edge_attr)
        x, edge_attr = self.relu3(x, edge_index, edge_attr)

        (node_o, edge_o), cluster_o = self.linear_o(x, edge_index, edge_attr), self.cluster_linear(x)
        node_o, edge_o, cluster_o = F.log_softmax(node_o, dim=-1), \
                                    F.log_softmax(edge_o, dim=-1), \
                                    F.log_softmax(cluster_o, dim=-1)

        return node_o, edge_o, cluster_o
        