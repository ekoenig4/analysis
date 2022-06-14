
import torch
from torch_geometric.data import Data
import torchmetrics.functional as f_metrics

from ..losses import *
from ..cpp_geometric import *
from .LightningModel import LightningModel

__all__ = [ 
    "SimplePair"
]

class SimpleClassifier(LightningModel):
    def __init__(self, loss='std_loss', **kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters('loss')
        self.f_loss = torch.nn.CrossEntropyLoss()

    def predict(self, data : Data, pair = None):
        with torch.no_grad():
            pair_o = self(data)
        return pair_o[:,1]

    def eval_batch(self, batch, pair_mask):
        batch.pair_mask = pair_mask == 1

        out = self(batch)
        y = batch.edge_y[batch.pair_mask]
        loss = self.f_loss(out, y)
        score = out[:,1]
        return score,loss

    def shared_step(self, batch, batch_idx, tag):

        pos_score, pos_loss = [], []
        for pos_mask in batch.pos_mask_index:
            score, loss = self.eval_batch(batch, pos_mask)
            pos_score.append(score)
            pos_loss.append(loss)
        pos_score = torch.cat(pos_score)
        pos_loss = torch.stack(pos_loss).mean()

        
        neg_score, neg_loss = [], []
        for neg_mask in batch.neg_mask_index:
            score, loss = self.eval_batch(batch, neg_mask)
            neg_score.append(score)
            neg_loss.append(loss)
        neg_score = torch.cat(neg_score)
        neg_loss = torch.stack(neg_loss).mean()

        loss = pos_loss + neg_loss

        cat_score = torch.cat([pos_score, neg_score])
        cat_truth = torch.cat([torch.ones(pos_score.shape), torch.zeros(neg_score.shape)]).long()
        auroc = f_metrics.auroc(cat_score, cat_truth)
        hitk1 = torch.mean(1.0*(pos_score.max(dim=0)[0] > neg_score.max(dim=0)[0]))


        metrics = dict(
            loss=loss, auroc=auroc, hitk1=hitk1
        )
        self.log_scalar(metrics, tag)

        histos = dict(
            pos_score=pos_score,
            neg_score=neg_score
        )
        self.log_histos(histos, tag)

        return metrics


class PairPredictor(torch.nn.Module):
    def __init__(self, n_in_node=None, n_in_edge=None, n_out=None):
        super().__init__()
        self.linear = torch.nn.Linear(2*n_in_node+n_in_edge, n_out)

    def forward(self, x, pair_index, pair_attr):
        x_i, x_j = x[pair_index]
        e_ij = torch.cat([ x_i, x_j, pair_attr], dim=-1 )
        return self.linear(e_ij)

class SimplePair(SimpleClassifier):
    name = "simple_pair"
    def __init__(self, nn_embed_1=64, nn_embed_2=128, nn_out_1=96, nn_out_2=32, **kwargs):
        super().__init__(**kwargs)

        self.node_embed = torch.nn.Linear(self.n_in_node, nn_embed_1)
        self.edge_embed = torch.nn.Linear(self.n_in_edge, nn_embed_1)
        self.pair_1 = PairPredictor(nn_embed_1, nn_embed_1, nn_embed_2)
        self.readout = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(nn_embed_2, nn_out_1),
            torch.nn.ReLU(),
            torch.nn.Linear(nn_out_1, nn_out_2),
            torch.nn.ReLU(),
            torch.nn.Linear(nn_out_2, 2),
            torch.nn.Softmax()
        )

    def forward(self, data : Data):
        data = data.to(self.device)
        x, edge_index, edge_attr, pair_mask = data.x, data.edge_index, data.edge_attr, data.pair_mask

        x = self.node_embed(x)
        edge_attr = self.edge_embed(edge_attr)
        x = self.pair_1(x, edge_index[:,pair_mask], edge_attr[pair_mask])
        x = self.readout(x)
        return x