
import pytorch_lightning as pl
import torch.nn.functional as F
from torchmetrics.functional import accuracy
import torch

from .gnn import to_tensor

from .losses import std_loss

class LightningModel(pl.LightningModule):
    def __init__(self, dataset, lr=1e-3, loss=std_loss):
        super().__init__()
        self.lr = lr
        self.loss = loss
        self.save_hyperparameters('lr', 'loss')
        self.hparams['node_attr_names'] = dataset.node_attr_names
        self.hparams['edge_attr_names'] = dataset.edge_attr_names
        
        
        self.node_weights = to_tensor(dataset.node_class_weights)
        self.edge_weights = to_tensor(dataset.edge_class_weights)
        self.type_weights = to_tensor(dataset.type_class_weights)

    def predict(self, data):
        with torch.no_grad():
            node_pred, edge_pred = self(data)
        return torch.exp(node_pred)[:, 1], torch.exp(edge_pred)[:, 1]

    def predict_nodes(self, data):
        node_pred, edge_pred = self.predict(data)
        return node_pred

    def predict_edges(self, data):
        node_pred, edge_pred = self.predict(data)
        return edge_pred
    

    def shared_step(self, batch, batch_idx, tag=None):
        node_o, edge_o = self(batch)

        loss = self.loss(self, node_o, edge_o, batch)
        
        acc = accuracy(torch.cat([node_o, edge_o]).argmax(
            dim=1), torch.cat([batch.y, batch.edge_y]))
        metrics = dict(loss=loss, acc=acc)
        if tag is not None:
            metrics = {f'{tag}_{key}': value for key, value in metrics.items()}

        self.log_dict(metrics)
        return metrics

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, tag='val')

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, tag='test')

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        return [optimizer], dict(
            scheduler=scheduler,
            monitor='val_loss'
        )