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
    "GoldenAttention"
]

class AttentionClassifier(LightningModel):
    def __init__(self, loss='std_loss', **kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters('loss')

    def predict(self, data : Data):
        with torch.no_grad():
            attention_o = self(data)
        return torch.exp(attention_o[:,1])

    def shared_step(self, batch, batch_idx, tag):

        metrics = dict()

        return metrics

class GoldenAttention(AttentionClassifier):
    name = "golden_attention"
    def __init__(self, nn_embed_1=64, nn_embed_2=128, nn_out_1=96, nn_out_2=32, **kwargs):
        super().__init__(**kwargs)

    def forward(self, data : Data):
        data = data.to(self.device)
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        