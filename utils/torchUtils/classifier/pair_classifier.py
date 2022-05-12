import torch
from torch import Tensor
from torch.nn import Module
import torch.nn.functional as F
from torch_geometric.data import Data
import torchmetrics.functional as f_metrics

from ..losses import *
from ..cpp_geometric import *
from .LightningModel import LightningModel
from ..gnn import k_max_neighbors, attr_undirected, top_accuracy, sample_pair
from torch_scatter import scatter_max

__all__ = [ 
    "GoldenPair",
]

class PairClassifier(LightningModel):
    def __init__(self, loss='std_loss', **kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters('loss')
        self.pair_loss = getattr(pair_losses, loss)

    def predict(self, data : Data):
        with torch.no_grad():
            pair_o = self(data)
        return torch.exp(pair_o[:,1])

    def shared_step(self, batch, batch_idx, tag):

        def _eval_pair(batch, pair):
            batch = sample_pair(batch, pair)
            o = self(batch)
            loss = self.pair_loss(self, o, batch)
            return o, loss

        pos_o, pos_loss = [], []
        for pos_pair in batch.pos_index:
            _pos_o, _pos_loss = _eval_pair(batch, pos_pair)
            pos_o.append(_pos_o)
            pos_loss.append(_pos_loss)
        pos_o = torch.stack(pos_o)
        pos_loss = torch.stack(pos_loss).mean()

        neg_o, neg_loss = [], []
        for neg_pair in batch.neg_index:
            _neg_o, _neg_loss = _eval_pair(batch, neg_pair)
            neg_o.append(_neg_o)
            neg_loss.append(_neg_loss)
        neg_o = torch.stack(neg_o)
        neg_loss = torch.stack(neg_loss).mean()

        pos_score, neg_score = torch.exp(pos_o[:,:,1]), torch.exp(neg_o[:,:,1])

        rank_loss = torch.mean(F.relu(1 - pos_score.unsqueeze(1) + neg_score.unsqueeze(0))**2)

        # loss = pos_loss + neg_loss
        loss = pos_loss + neg_loss + rank_loss
        # loss = rank_loss
        hits = torch.mean( 1.0*(pos_score>neg_score.max(dim=0)[0]) )
        maxhits = torch.mean( 1.0*(pos_score.max(dim=0)[0]>neg_score.max(dim=0)[0]) )

        metrics = dict(
            loss=loss,hits=hits,maxhits=maxhits
        )
        self.log_scalar(metrics, tag)

        histos = dict(
            pos_score=pos_score.reshape(-1),
            neg_score=neg_score.reshape(-1)
        )
        self.log_histos(histos, tag)

        return metrics


class PairPredictor(torch.nn.Module):
    def __init__(self, n_in_node=None, n_out=None):
        super().__init__()
        self.linear = torch.nn.Linear(2*n_in_node, n_out)

    def forward(self, x, pair_index):
        x_i, x_j = x[pair_index]
        e_ij = torch.cat([ torch.abs(x_i + x_j), torch.abs(x_i - x_j)], dim=-1 )
        return self.linear(e_ij)

class GoldenPair(PairClassifier):
    name = "golden_pair"
    def __init__(self, nn_embed_1=64, nn_embed_2=128, nn_out_1=96, nn_out_2=32, **kwargs):
        super().__init__(**kwargs)
        self.embed_1 = layers.GCNLinear(self.n_in_node+1, self.n_in_edge, nn_embed_1)
        self.conv_1 = layers.GCNConvMask(n_in_node=nn_embed_1, n_in_edge=nn_embed_1, n_out=nn_embed_1)
        self.norm_1 = layers.GCNBatchNorm(nn_embed_1, nn_embed_1)
        self.relu_1 = layers.GCNRelu()
        
        self.embed_2 = layers.GCNLinear(nn_embed_1, nn_embed_1, nn_embed_2)
        self.conv_2 = layers.GCNConvMask(n_in_node=nn_embed_2, n_in_edge=nn_embed_2, n_out=nn_embed_2)
        self.norm_2 = layers.GCNBatchNorm(nn_embed_2, nn_embed_2)
        self.relu_2 = layers.GCNRelu()

        self.pair_1 = PairPredictor(nn_embed_2, nn_out_1)
        self.relu_3 = torch.nn.ReLU()

        self.readout = torch.nn.Sequential(
            torch.nn.Linear(nn_out_1, nn_out_2),
            torch.nn.ReLU(),
            torch.nn.Linear(nn_out_2, 2),
            torch.nn.LogSoftmax(dim=-1)
        )

    def forward(self, data : Data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        edge_mask, batch = data.get('edge_mask', None), data.get('batch', None)

        x, edge_attr = self.embed_1(x, edge_index, edge_attr)
        x, _ = self.conv_1(x, edge_index, edge_attr, edge_mask)        
        x, _ = self.norm_1(x, edge_index, edge_attr, batch)
        x, _ = self.relu_1(x, edge_index, edge_attr)        

        x, edge_attr = self.embed_2(x, edge_index, edge_attr)
        x, _ = self.conv_2(x, edge_index, edge_attr, edge_mask)
        x, _ = self.norm_2(x, edge_index, edge_attr, batch)
        x, _ = self.relu_2(x, edge_index, edge_attr)

        pair_x = self.pair_1(x, data.pair_index)
        pair_x = self.relu_3(pair_x)
        pair_x = self.readout(pair_x)

        return pair_x
