from psutil import net_io_counters
import torch
from torch import Tensor
from torch.nn import Module
import torch.nn.functional as F
from torch_geometric.data import Data
import torchmetrics.functional as f_metrics

from ..losses import *
from ..cpp_geometric import *
from .LightningModel import LightningModel
from ..gnn import k_max_neighbors, attr_undirected, top_accuracy
from torch_scatter import scatter_max

__all__ = [ 
    "GoldenEdge", "GoldenEdgeV2", "GoldenGCN", "GoldenKNN","GoldenKNN_v2"
]

class EdgeClassifier(LightningModel):
    def __init__(self, loss='std_loss', **kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters('loss')
        # self.node_loss = getattr(node_losses, loss)
        self.edge_loss = getattr(edge_losses, loss)

    def predict(self, data : Data):
        with torch.no_grad():
            node_o, edge_o = self(data)
        return torch.exp(node_o[:,1]), torch.exp(edge_o[:,1])

    def shared_step(self, batch, batch_idx, tag):
        node_o, edge_o = self(batch)
        # loss = self.node_loss(self, node_o, batch) + self.edge_loss(self, edge_o, batch)
        loss = self.edge_loss(self, edge_o, batch)

        node_score = torch.exp(node_o[:,1])
        node_acc   = f_metrics.accuracy(node_score, batch.y)
        node_auroc = f_metrics.auroc(node_score, batch.y)

        edge_score = torch.exp(edge_o[:,1])
        edge_acc   = f_metrics.accuracy(edge_score, batch.edge_y)
        edge_auroc = f_metrics.auroc(edge_score, batch.edge_y)
        top_acc    = top_accuracy(edge_score, batch)

        # kmax2_mask = k_max_neighbors(edge_score, batch.edge_index, n_neighbor=2, remove_self=True)
        # kmax2_score, kmax2_y = edge_score[kmax2_mask], batch.edge_y[kmax2_mask]

        # kmax2_acc = f_metrics.accuracy(kmax2_score, kmax2_y)
        # kmax2_auroc = f_metrics.auroc(kmax2_score, kmax2_y)

        metrics = dict(
            loss=loss, 
            node_acc=node_acc, node_auroc=node_auroc,
            edge_acc=edge_acc, edge_auroc=edge_auroc,
            top_acc=top_acc,
            # kmax2_acc=kmax2_acc, kmax2_auroc=kmax2_auroc,
        )
        self.log_scalar(metrics, tag)

        true_node_score = node_score[batch.y == 1]
        fake_node_score = node_score[batch.y == 0]

        true_edge_score = edge_score[batch.edge_y == 1]
        fake_edge_score = edge_score[batch.edge_y == 0]
        
        # true_kmax2_score = kmax2_score[kmax2_y == 1]
        # fake_kmax2_score = kmax2_score[kmax2_y == 0]

        histos = dict(
            true_node_score=true_node_score, fake_node_score=fake_node_score,
            true_edge_score=true_edge_score, fake_edge_score=fake_edge_score,
            # true_kmax2_score=true_kmax2_score, fake_kmax2_score=fake_kmax2_score,
        )
        self.log_histos(histos, tag)

        return metrics


class GoldenEdge(EdgeClassifier):
    name = 'golden_edge'
    def __init__(self, nn_conv_1=32, nn_linear_1=96, nn_linear_2=64, **kwargs):
        super().__init__(**kwargs)

        self.conv = layers.GCNConvMSG(2*self.n_in_node+self.n_in_edge, nn_conv_1)
        self.linear_1 = layers.GCNLinear(nn_conv_1, nn_conv_1, nn_linear_1)

        self.relu = layers.GCNRelu()
        self.norm = layers.GCNNormalize()

        self.linear_2 = layers.GCNLinear(nn_linear_1, nn_linear_1, nn_linear_2)
        self.linear_o = layers.GCNLinear(nn_linear_2, nn_linear_2, 2)
        self.log_softmax = layers.GCNLogSoftmax()

    def forward(self, data : Data) -> Tensor:
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x, edge_attr = self.conv(x, edge_index, edge_attr)
        x, edge_attr = self.relu(x, edge_index, edge_attr)
        # x, edge_attr = self.norm(x, edge_index, edge_attr)
        x, edge_attr = self.linear_1(x, edge_index, edge_attr)
        x, edge_attr = self.relu(x, edge_index, edge_attr)
        
        x, edge_attr = self.linear_2(x, edge_index, edge_attr)
        x, edge_attr = self.relu(x, edge_index, edge_attr)
        
        x, edge_attr = self.linear_o(x, edge_index, edge_attr)
        x, edge_attr = self.relu(x, edge_index, edge_attr)

        x, edge_attr = self.log_softmax(x, edge_index, edge_attr)
        return x, edge_attr
        

class GoldenEdgeV2(EdgeClassifier):
    name = 'golden_edge_v2'
    def __init__(self, nn_conv_1=32, nn_linear_1=64, nn_conv_2=96, nn_linear_2=128, nn_linear_3=96, **kwargs):
        super().__init__(**kwargs)

        self.conv_1 = layers.GCNConvMSG(2*self.n_in_node+self.n_in_edge, nn_conv_1)
        self.linear_1 = layers.GCNLinear(nn_conv_1, nn_conv_1, nn_linear_1)

        self.conv_2 = layers.GCNConvMSG(3*nn_linear_1, nn_conv_2)
        self.linear_2 = layers.GCNLinear(nn_conv_2, nn_conv_2, nn_linear_2)

        self.relu = layers.GCNRelu()
        self.norm = layers.GCNNormalize()

        self.linear_3 = layers.GCNLinear(nn_linear_2, nn_linear_2, nn_linear_3)

        self.linear_o = layers.GCNLinear(nn_linear_3, nn_linear_3, 2)
        self.log_softmax = layers.GCNLogSoftmax()

    def forward(self, data : Data) -> Tensor:
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x, edge_attr = self.conv_1(x, edge_index, edge_attr)
        x, edge_attr = self.relu(x, edge_index, edge_attr)
        # x, edge_attr = self.norm(x, edge_index, edge_attr)
        x, edge_attr = self.linear_1(x, edge_index, edge_attr)
        x, edge_attr = self.relu(x, edge_index, edge_attr)
        
        x, edge_attr = self.conv_2(x, edge_index, edge_attr)
        x, edge_attr = self.relu(x, edge_index, edge_attr)
        # x, edge_attr = self.norm(x, edge_index, edge_attr)
        x, edge_attr = self.linear_2(x, edge_index, edge_attr)
        x, edge_attr = self.relu(x, edge_index, edge_attr)

        x, edge_attr = self.linear_3(x, edge_index, edge_attr)
        x, edge_attr = self.relu(x, edge_index, edge_attr)
        
        x, edge_attr = self.linear_o(x, edge_index, edge_attr)
        x, edge_attr = self.relu(x, edge_index, edge_attr)

        x, edge_attr = self.log_softmax(x, edge_index, edge_attr)
        return x, edge_attr

class GoldenGCN(EdgeClassifier):
    name = "golden_gcn"
    def __init__(self, nn_conv1_out=32, nn_conv2_out=128, nn_linear_out=64, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = layers.GCNConvMSG(n_in_node=self.n_in_node, n_in_edge=self.n_in_edge, n_out=nn_conv1_out)
        self.relu1 = layers.GCNRelu()
        self.conv2 = layers.GCNConvMSG(n_in_node=nn_conv1_out, n_in_edge=nn_conv1_out, n_out=nn_conv2_out)
        self.relu2 = layers.GCNRelu()

        self.linear1 = layers.GCNLinear(nn_conv2_out, nn_conv2_out, nn_linear_out)
        
        self.relu3 = layers.GCNRelu()
        
        self.linear_o = layers.GCNLinear(nn_linear_out, nn_linear_out, 2)
        self.log_softmax = layers.GCNLogSoftmax()
        
    def forward(self, data : Data) -> Tensor:
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x, edge_attr = self.conv1(x, edge_index, edge_attr)
        x, edge_attr = self.relu1(x, edge_index, edge_attr)

        x, edge_attr = self.conv2(x, edge_index, edge_attr)
        x, edge_attr = self.relu2(x, edge_index, edge_attr)

        x, edge_attr = self.linear1(x, edge_index, edge_attr)
        x, edge_attr = self.relu3(x, edge_index, edge_attr)

        x, edge_attr = self.linear_o(x, edge_index, edge_attr)
        x, edge_attr = self.log_softmax(x, edge_index, edge_attr)

        return x, edge_attr

class UndirectedEdges(torch.nn.Module):
    def __init__(self, aggr='max'):
        super().__init__()
        self.aggr = aggr

    def forward(self, edge_index, edge_attr):
        return attr_undirected(edge_index, edge_attr, self.aggr)

class GoldenKNN(EdgeClassifier):
    name = "golden_knn"
    def __init__(self, nn_embed_1=64, nn_embed_2=128, nn_out_1=96, nn_out_2=32, **kwargs):
        super().__init__(**kwargs)
        self.embed_1 = layers.GCNLinear(self.n_in_node, self.n_in_edge, nn_embed_1)
        self.conv_1 = layers.GCNConvMask(n_in_node=nn_embed_1, n_in_edge=nn_embed_1, n_out=nn_embed_1)
        self.norm_1 = layers.GCNBatchNorm(nn_embed_1, nn_embed_1)
        self.relu_1 = layers.GCNRelu()
        
        self.embed_2 = layers.GCNLinear(nn_embed_1, nn_embed_1, nn_embed_2)
        self.conv_2 = layers.GCNConvMask(n_in_node=nn_embed_2, n_in_edge=nn_embed_2, n_out=nn_embed_2)
        self.norm_2 = layers.GCNBatchNorm(nn_embed_2, nn_embed_2)
        self.relu_2 = layers.GCNRelu()

        self.node_readout = torch.nn.Sequential(
            torch.nn.Linear(nn_embed_2, nn_out_1),
            torch.nn.ReLU(),
            torch.nn.Linear(nn_out_1, nn_out_2),
            torch.nn.ReLU(),
            torch.nn.Linear(nn_out_2, 2),
        )
        
        self.edge_readout = torch.nn.Sequential(
            torch.nn.Linear(nn_embed_2, nn_out_1),
            torch.nn.ReLU(),
            torch.nn.Linear(nn_out_1, nn_out_2),
            torch.nn.ReLU(),
            torch.nn.Linear(nn_out_2, 2),
        )

        self.undirected = UndirectedEdges('max')
        self.log_softmax = layers.GCNLogSoftmax()

    def forward(self, data : Data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        edge_mask, batch = data.get('edge_mask', None), data.get('batch', None)

        x, edge_attr = self.embed_1(x, edge_index, edge_attr)
        x, edge_attr = self.conv_1(x, edge_index, edge_attr, edge_mask)        
        x, edge_attr = self.norm_1(x, edge_index, edge_attr, batch)
        x, edge_attr = self.relu_1(x, edge_index, edge_attr)        

        x, edge_attr = self.embed_2(x, edge_index, edge_attr)
        x, edge_attr = self.conv_2(x, edge_index, edge_attr, edge_mask)
        x, edge_attr = self.norm_2(x, edge_index, edge_attr, batch)
        x, edge_attr = self.relu_2(x, edge_index, edge_attr)

        x, edge_attr = self.node_readout(x), self.edge_readout(edge_attr)
        edge_attr = self.undirected(edge_index, edge_attr)
        x, edge_attr = self.log_softmax(x, edge_index, edge_attr)

        return x, edge_attr


class GoldenKNN_v2(EdgeClassifier):
    name = "golden_knn_v2"
    def __init__(self, nn_embed_1=64, nn_embed_2=128, nn_out_1=96, nn_out_2=32, **kwargs):
        super().__init__(**kwargs)
        self.embed_1 = layers.GCNLinear(self.n_in_node, self.n_in_edge, nn_embed_1)
        self.conv_1 = layers.GCNConvMask(n_in_node=nn_embed_1, n_in_edge=nn_embed_1, n_out=nn_embed_1)
        self.relu_1 = layers.GCNRelu()
        
        self.embed_2 = layers.GCNLinear(nn_embed_1, nn_embed_1, nn_embed_2)
        self.conv_2 = layers.GCNConvMask(n_in_node=nn_embed_2, n_in_edge=nn_embed_2, n_out=nn_embed_2)
        self.relu_2 = layers.GCNRelu()

        self.conv_3 = layers.GCNConvMSG(n_in_node=nn_embed_2, n_in_edge=nn_embed_2, n_out=nn_out_1)
        self.node_readout = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(nn_out_1, nn_out_2),
            torch.nn.ReLU(),
            torch.nn.Linear(nn_out_2, 2),
        )
        
        self.edge_readout = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(nn_out_1, nn_out_2),
            torch.nn.ReLU(),
            torch.nn.Linear(nn_out_2, 2),
        )

        # self.undirected = UndirectedEdges('max')
        self.log_softmax = layers.GCNLogSoftmax()

    def forward(self, data : Data):
        x, edge_index, edge_attr, edge_mask = data.x, data.edge_index, data.edge_attr, data.get('edge_mask', None)

        x, edge_attr = self.embed_1(x, edge_index, edge_attr)
        x, edge_attr = self.conv_1(x, edge_index, edge_attr, edge_mask)        
        x, edge_attr = self.relu_1(x, edge_index, edge_attr)        

        x, edge_attr = self.embed_2(x, edge_index, edge_attr)
        x, edge_attr = self.conv_2(x, edge_index, edge_attr, edge_mask)
        x, edge_attr = self.relu_2(x, edge_index, edge_attr)

        x, edge_attr = self.conv_3(x, edge_index, edge_attr)
        x, edge_attr = self.node_readout(x), self.edge_readout(edge_attr)
        # edge_attr = self.undirected(edge_index, edge_attr)
        x, edge_attr = self.log_softmax(x, edge_index, edge_attr)

        return x, edge_attr
