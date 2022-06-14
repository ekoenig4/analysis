
import torch
from torch.nn.functional import relu, binary_cross_entropy
from torch_geometric.data import Data
import torchmetrics.functional as f_metrics

from ..losses import *
from ..cpp_geometric import *
from .LightningModel import LightningModel

__all__ = [ 
    "GoldenQuadH"
]

class QuadHClassifier(LightningModel):
    def __init__(self, loss='std_loss', **kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters('loss')

    def predict(self, data : Data, pair = None):
        with torch.no_grad():
            pair_o = self(data)
        return pair_o[:,1]

    def eval_batch(self, batch, quad_mask):
        batch.quad_mask = quad_mask
        score = self(batch)[:,0]    
        # loss = quadH_losses.mass_spread_loss(batch, quad_mask)
        # return [score, loss]
        return score

    def shared_step(self, batch, batch_idx, tag):

        scores = torch.cat([ self.eval_batch(batch, quad_mask) for quad_mask in batch.sampled_quad_mask.T ])
        ranks = torch.unique(batch.sampled_quad_y)
        rank_masks = [ batch.sampled_quad_y == rank for rank in ranks ]
        y = (1*(batch.sampled_quad_y == 4)).long()

        loss = binary_cross_entropy(scores, y.float())
        for mask_lo, mask_hi in zip(rank_masks[:-1], rank_masks[1:]):
            loss = loss + torch.mean(relu(1 - scores[mask_hi] + scores[mask_lo]))

        auroc = f_metrics.auroc(scores, y)

        metrics = dict(loss=loss, auroc=auroc)
        self.log_scalar(metrics, tag)

        return metrics


class PairPredictor(torch.nn.Module):
    def __init__(self, n_in_node=None, n_in_edge=None, n_out=None):
        super().__init__()
        self.linear = torch.nn.Linear(2*n_in_node+n_in_edge, n_out)

    def forward(self, x, pair_index, pair_attr):
        x_i, x_j = x[pair_index]
        e_ij = torch.cat([ x_i, x_j, pair_attr], dim=-1 )
        return self.linear(e_ij)

class QuadPredictor(torch.nn.Module):
    def __init__(self, n_in_node=None, n_in_edge=None, n_out=None, aggr='max'):
        super().__init__()
        self.aggr = aggr

    def forward(self, x):
        quad_x = [ x[i::4] for i in range(4) ]

        if self.aggr == 'cat':
            x = torch.cat(quad_x, dim=-1)
        elif self.aggr == 'max':
            x = torch.cat([quad.unsqueeze(-1) for quad in quad_x],dim=-1)
            x = x.max(dim=-1)[0]

        return x
        

class GoldenQuadH(QuadHClassifier):
    name = "golden_quadh"
    def __init__(self, nn_embed_1=32, nn_embed_2=64, nn_out_1=96, nn_out_2=32, **kwargs):
        super().__init__(**kwargs)

        # self.node_embed = torch.nn.Linear(self.n_in_node, nn_embed_1)
        # self.edge_embed = torch.nn.Linear(self.n_in_edge, nn_embed_1)

        self.gnn_module = layers.GCNConvMask(n_in_node=self.n_in_node, n_in_edge=self.n_in_edge, n_out=nn_embed_1)
        self.gnn_norm = layers.BatchNorm(nn_embed_1)
        self.gnn_relu = torch.nn.ReLU()

        self.pair_module = PairPredictor(nn_embed_1, self.n_in_edge, nn_embed_2)
        self.pair_norm = layers.BatchNorm(nn_embed_2)
        self.pair_relu = torch.nn.ReLU()

        self.quad_module = QuadPredictor(aggr="cat")
        self.readout = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(4*nn_embed_2, nn_out_1),
            torch.nn.ReLU(),
            torch.nn.Linear(nn_out_1, nn_out_2),
            torch.nn.ReLU(),
            torch.nn.Linear(nn_out_2, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, data : Data):
        data = data.to(self.device)
        x, edge_index, edge_attr, quad_mask = data.x, data.edge_index, data.edge_attr, data.quad_mask
        edge_mask, quad_mask = data.edge_mask, data.quad_mask

        batch = data.get('batch',None)
        x,_ = self.gnn_module(x, edge_index[:,edge_mask], edge_attr[edge_mask])
        x = self.gnn_norm(x, batch)
        x = self.gnn_relu(x)

        if batch is not None: batch = batch[ edge_index[:,quad_mask] ][0]
        x = self.pair_module(x, edge_index[:,quad_mask], edge_attr[quad_mask])
        x = self.pair_norm(x, batch)
        x = self.pair_relu(x)

        x = self.quad_module(x)

        x = self.readout(x)
        return x