import torch
from torch import Tensor
from torch.nn import Module
import torch.nn.functional as F
from torch_geometric.data import Data
import torchmetrics.functional as f_metrics

from ..losses import *
from ..cpp_geometric import *
from .LightningModel import LightningModel

__all__ = [ 
    "GoldenNode", "GoldenNodeV2"
]

class NodeClassifier(LightningModel):
    def __init__(self, loss='std_loss', **kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters('loss')
        self.loss = getattr(node_losses, loss)
    def shared_step(self, batch, batch_idx, tag):
        node_o = self(batch)
        loss = self.loss(self, node_o, batch)

        node_score = torch.exp(node_o[:,1])
        node_acc   = f_metrics.accuracy(node_score, batch.y)
        node_auroc = f_metrics.auroc(node_score, batch.y)

        metrics = dict(
            loss=loss, 
            node_acc=node_acc, node_auroc=node_auroc,
        )
        self.log_scalar(metrics, tag)

        true_node_score = node_score[batch.y == 1]
        fake_node_score = node_score[batch.y == 0]
        
        histos = dict(
            true_node_score=true_node_score, fake_node_score=fake_node_score
        )
        self.log_histos(histos, tag)

        return metrics

class GoldenNode(NodeClassifier):
    name = 'golden_node'
    def __init__(self, nn_conv_1=32, nn_linear_1=96, nn_linear_2=64, **kwargs):
        super().__init__(**kwargs)

        self.conv = layers.GCNConvMSG(2*self.n_in_node+self.n_in_edge, nn_conv_1)
        self.linear_1 = layers.NodeLinear(nn_conv_1, nn_linear_1)

        self.relu = layers.GCNRelu()
        self.norm = layers.GCNNormalize()

        self.linear_2 = layers.NodeLinear(nn_linear_1, nn_linear_2)
        self.linear_o = layers.NodeLinear(nn_linear_2, 2)
        self.log_softmax = layers.GCNLogSoftmax()

    def forward(self, data : Data) -> Tensor:
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x, edge_attr = self.conv(x, edge_index, edge_attr)
        x, edge_attr = self.relu(x, edge_index, edge_attr)
        x, edge_attr = self.norm(x, edge_index, edge_attr)
        x, edge_attr = self.linear_1(x, edge_index, edge_attr)
        x, edge_attr = self.relu(x, edge_index, edge_attr)
        
        x, edge_attr = self.linear_2(x, edge_index, edge_attr)
        x, edge_attr = self.relu(x, edge_index, edge_attr)
        
        x, edge_attr = self.linear_o(x, edge_index, edge_attr)
        x, edge_attr = self.relu(x, edge_index, edge_attr)

        x, edge_attr = self.log_softmax(x, edge_index, edge_attr)
        return x


class GoldenNodeV2(NodeClassifier):
    name = 'golden_node_v2'
    def __init__(self, nn_conv_1=32, nn_linear_1=64, nn_conv_2=96, nn_linear_2=128, nn_linear_3=96, **kwargs):
        super().__init__(**kwargs)

        self.conv_1 = layers.GCNConvMSG(2*self.n_in_node+self.n_in_edge, nn_conv_1)
        self.linear_1 = layers.NodeLinear(nn_conv_1, nn_linear_1)

        self.conv_2 = layers.GCNConvMSG(3*nn_linear_1, nn_conv_2)
        self.linear_2 = layers.NodeLinear(nn_conv_2, nn_linear_2)

        self.relu = layers.GCNRelu()
        self.norm = layers.GCNNormalize()

        self.linear_3 = layers.NodeLinear(nn_linear_2, nn_linear_3)

        self.linear_o = layers.NodeLinear(nn_linear_3, 2)
        self.log_softmax = layers.GCNLogSoftmax()

    def forward(self, data : Data) -> Tensor:
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x, edge_attr = self.conv_1(x, edge_index, edge_attr)
        x, edge_attr = self.relu(x, edge_index, edge_attr)
        x, edge_attr = self.norm(x, edge_index, edge_attr)
        x, edge_attr = self.linear_1(x, edge_index, edge_attr)
        x, edge_attr = self.relu(x, edge_index, edge_attr)
        
        x, edge_attr = self.conv_2(x, edge_index, edge_attr)
        x, edge_attr = self.relu(x, edge_index, edge_attr)
        x, edge_attr = self.norm(x, edge_index, edge_attr)
        x, edge_attr = self.linear_2(x, edge_index, edge_attr)
        x, edge_attr = self.relu(x, edge_index, edge_attr)

        x, edge_attr = self.linear_3(x, edge_index, edge_attr)
        x, edge_attr = self.relu(x, edge_index, edge_attr)
        
        x, edge_attr = self.linear_o(x, edge_index, edge_attr)
        x, edge_attr = self.relu(x, edge_index, edge_attr)

        x, edge_attr = self.log_softmax(x, edge_index, edge_attr)
        return x

