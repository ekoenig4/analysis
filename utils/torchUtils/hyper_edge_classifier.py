from sympy import re
import torch
from torch import Tensor
from torch.nn import Module
import torch.nn.functional as F
from torch_geometric.data import Data
import torchmetrics.functional as f_metrics

from . import edge_losses, node_losses, hyper_edge_losses
from .cpp_geometric import *
from .LightningModel import LightningModel

__all__ = [ 
    "GoldenHyperEdge", "GoldenHyperEdgeConv"
]

class HyperEdgeClassifier(LightningModel):
    def __init__(self, loss='std_loss', **kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters('loss')
        self.node_loss = getattr(node_losses, loss)
        self.edge_loss = getattr(edge_losses, loss)
        self.hyper_edge_loss = getattr(hyper_edge_losses,loss)

    def shared_step(self, batch, batch_idx, tag):
        node_o, edge_o, hyper_edge_o = self(batch)
        loss = self.node_loss(self, node_o, batch) + self.edge_loss(self, edge_o, batch) + self.hyper_edge_loss(self, hyper_edge_o, batch)
        # loss = self.edge_loss(self, edge_o, batch) + self.hyper_edge_loss(self, hyper_edge_o, batch)
        # loss = self.hyper_edge_loss(self, hyper_edge_o, batch)

        node_score = torch.exp(node_o[:,1])
        node_auroc = f_metrics.auroc(node_score, batch.y)

        edge_score = torch.exp(edge_o[:,1])
        edge_auroc = f_metrics.auroc(edge_score, batch.edge_y)

        hyper_edge_score = torch.exp(hyper_edge_o[:,1])
        hyper_edge_auroc = f_metrics.auroc(hyper_edge_score, 1*(batch.hyper_edge_y == 1))

        metrics = dict(
            loss=loss, 
            node_auroc=node_auroc,
            edge_auroc=edge_auroc,
            hyper_edge_auroc=hyper_edge_auroc,
        )
        self.log_scalar(metrics, tag)

        true_node_score = node_score[batch.y == 1]
        fake_node_score = node_score[batch.y == 0]

        true_edge_score = edge_score[batch.edge_y == 1]
        fake_edge_score = edge_score[batch.edge_y == 0]

        true_hyper_edge_score = hyper_edge_score[batch.hyper_edge_y == 1]
        fake_hyper_edge_score = hyper_edge_score[batch.hyper_edge_y != 1]

        histos = dict(
            true_node_score=true_node_score, fake_node_score=fake_node_score,
            true_edge_score=true_edge_score, fake_edge_score=fake_edge_score,
            true_hyper_edge_score  =true_hyper_edge_score  , fake_hyper_edge_score  =fake_hyper_edge_score,
        )
        self.log_histos(histos, tag)

        return metrics

class GoldenHyperEdge(HyperEdgeClassifier):
    name = 'golden_hyper_edge'
    def __init__(self, nn_conv1_out=32, nn_conv2_out=64, nn_linear1_out=128, nn_linear2_out=96, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = layers.GCNConvMSG(n_in_node=self.n_in_node, n_in_edge=self.n_in_edge, n_out=nn_conv1_out)
        self.relu1 = layers.GCNRelu()
        self.conv2 = layers.GCNConvMSG(n_in_node=nn_conv1_out, n_in_edge=nn_conv1_out, n_out=nn_conv2_out)
        self.relu2 = layers.GCNRelu()

        self.hyper_linear_1 = layers.HyperEdgeLinear(nn_conv2_out, nn_linear1_out)
        self.hyper_linear_2 = torch.nn.Linear(nn_linear1_out, nn_linear2_out)
        self.hyper_linear_o = torch.nn.Linear(nn_linear2_out, 2)
        
        self.edge_linear_1 = torch.nn.Linear(nn_conv2_out, nn_linear1_out)
        self.edge_linear_2 = torch.nn.Linear(nn_linear1_out, nn_linear2_out)
        self.edge_linear_o = torch.nn.Linear(nn_linear2_out, 2)

        self.node_linear_1 = torch.nn.Linear(nn_conv2_out, nn_linear1_out)
        self.node_linear_2 = torch.nn.Linear(nn_linear1_out, nn_linear2_out)
        self.node_linear_o = torch.nn.Linear(nn_linear2_out, 2)

    def forward(self, data : Data) -> Tensor:
        x, edge_index, edge_attr, hyper_edge_index = data.x, data.edge_index, data.edge_attr, data.hyper_edge_index

        x, edge_attr = self.conv1(x, edge_index, edge_attr)
        x, edge_attr = self.relu1(x, edge_index, edge_attr)

        x, edge_attr = self.conv2(x, edge_index, edge_attr)
        x, edge_attr = self.relu2(x, edge_index, edge_attr)

        hyper_edge_attr = self.hyper_linear_1(x, hyper_edge_index)
        hyper_edge_attr = F.relu(hyper_edge_attr)
        
        hyper_edge_attr = self.hyper_linear_2(hyper_edge_attr)
        hyper_edge_attr = F.relu(hyper_edge_attr)

        hyper_edge_attr = self.hyper_linear_o(hyper_edge_attr)
        hyper_edge_attr = F.log_softmax(hyper_edge_attr, dim=-1)

        edge_attr = self.edge_linear_1(edge_attr)
        edge_attr = F.relu(edge_attr)
        
        edge_attr = self.edge_linear_2(edge_attr)
        edge_attr = F.relu(edge_attr)

        edge_attr = self.edge_linear_o(edge_attr)
        edge_attr = F.log_softmax(edge_attr, dim=-1)

        x = self.node_linear_1(x)
        x = F.relu(x)
        
        x = self.node_linear_2(x)
        x = F.relu(x)
        
        x = self.node_linear_o(x)
        x = F.log_softmax(x, dim=-1)

        return x, edge_attr, hyper_edge_attr
 

class GraphRelu(Module):
    def forward(self, *attrs):
        return [ F.relu(attr) for attr in attrs ]

class GraphLogSoftmax(Module):
    def forward(self, *attrs):
        return [ F.log_softmax(attr, dim=-1) for attr in attrs ]
        
class GraphNormalize(Module):
    def forward(self, *attrs):
        return [ F.normalize(attr, 2) for attr in attrs ]

class GoldenHyperEdgeConv(HyperEdgeClassifier):
    name = "golden_hyper_edge_conv"
    def __init__(self, nn_conv_1=64, nn_conv_2=128, nn_linear_1=256, nn_linear_2=96, **kwargs):
        super().__init__(**kwargs)
        
        self.edge_conv_1 = layers.GCNConvMSG(n_in_node=self.n_in_node, n_in_edge=self.n_in_edge, n_out=nn_conv_1)
        self.hype_conv_1 = layers.HyperEdgeConvMSG(n_in_node=nn_conv_1, n_in_hyper=1, n_out=2*nn_conv_1)
        self.g_relu_1 = GraphRelu()
        self.g_norm_1 = GraphNormalize()

        self.edge_conv_2 = layers.GCNConvMSG(n_in_node=2*nn_conv_1, n_in_edge=nn_conv_1, n_out=nn_conv_2)
        self.hype_conv_2 = layers.HyperEdgeConvMSG(n_in_node=nn_conv_2, n_in_hyper=2*nn_conv_1, n_out=2*nn_conv_2)
        self.g_relu_2 = GraphRelu()
        self.g_norm_2 = GraphNormalize()

        self.node_linear_1 = torch.nn.Linear(2*nn_conv_2, nn_linear_1)
        self.edge_linear_1 = torch.nn.Linear(  nn_conv_2, nn_linear_1)
        self.hype_linear_1 = torch.nn.Linear(2*nn_conv_2, nn_linear_1)
        self.g_relu_3 = GraphRelu()

        self.node_linear_2 = torch.nn.Linear(nn_linear_1, nn_linear_2)
        self.edge_linear_2 = torch.nn.Linear(nn_linear_1, nn_linear_2)
        self.hype_linear_2 = torch.nn.Linear(nn_linear_1, nn_linear_2)
        self.g_relu_4 = GraphRelu()

        self.node_linear_o = torch.nn.Linear(nn_linear_2, 2)
        self.edge_linear_o = torch.nn.Linear(nn_linear_2, 2)
        self.hype_linear_o = torch.nn.Linear(nn_linear_2, 2)
        self.log_softmax = GraphLogSoftmax()

    def forward(self, data : Data) -> Tensor:
        x, edge_index, edge_attr, hyper_edge_index, hyper_edge_attr = data.x, \
            data.edge_index, data.edge_attr, \
            data.hyper_edge_index, data.hyper_edge_attr

        x, edge_attr = self.edge_conv_1(x, edge_index, edge_attr)
        x, edge_attr = self.g_relu_1(x, edge_attr)
        x, hyper_edge_attr = self.hype_conv_1(x, hyper_edge_index, hyper_edge_attr)
        x, hyper_edge_attr = self.g_relu_1(x, hyper_edge_attr)
        x, edge_attr, hyper_edge_attr = self.g_norm_1(x, edge_attr, hyper_edge_attr)

        x, edge_attr = self.edge_conv_2(x, edge_index, edge_attr)
        x, edge_attr = self.g_relu_2(x, edge_attr)
        x, hyper_edge_attr = self.hype_conv_2(x, hyper_edge_index, hyper_edge_attr)
        x, hyper_edge_attr = self.g_relu_2(x, hyper_edge_attr)
        x, edge_attr, hyper_edge_attr = self.g_norm_2(x, edge_attr, hyper_edge_attr)

        x, edge_attr, hyper_edge_attr = self.node_linear_1(x), self.edge_linear_1(edge_attr), self.hype_linear_1(hyper_edge_attr)
        x, edge_attr, hyper_edge_attr = self.g_relu_3(x, edge_attr, hyper_edge_attr)

        x, edge_attr, hyper_edge_attr = self.node_linear_2(x), self.edge_linear_2(edge_attr), self.hype_linear_2(hyper_edge_attr)
        x, edge_attr, hyper_edge_attr = self.g_relu_3(x, edge_attr, hyper_edge_attr)

        x, edge_attr, hyper_edge_attr = self.node_linear_o(x), self.edge_linear_o(edge_attr), self.hype_linear_o(hyper_edge_attr)
        x, edge_attr, hyper_edge_attr = self.log_softmax(x, edge_attr, hyper_edge_attr)
        return x, edge_attr, hyper_edge_attr

