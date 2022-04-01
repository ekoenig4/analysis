
import pytorch_lightning as pl
import torch.nn.functional as F
from torchmetrics.functional import accuracy, auroc
import torch

from .gnn import to_tensor, config

from . import losses

class LightningModel(pl.LightningModule):
    def __init__(self, dataset=None, lr=1e-3, loss='std_loss', batch_size=None, node_attr_names=None, edge_attr_names=None, scale=None, **kwargs):
        super().__init__()
        self.lr = lr
        
        if dataset is not None:
            scale = dataset.scale 
            node_attr_names = dataset.node_attr_names
            edge_attr_names = dataset.edge_attr_names
            
        self.save_hyperparameters('lr', 'loss','batch_size')
        self.hparams['scale'] = scale
        self.hparams['node_attr_names'] = node_attr_names
        self.hparams['edge_attr_names'] = edge_attr_names
        
        self.loss = losses.lossMap[loss]
        
        if dataset is not None:
            self.node_weights = to_tensor(dataset.node_class_weights)
            self.edge_weights = to_tensor(dataset.edge_class_weights)
            self.type_weights = to_tensor(dataset.type_class_weights)

    def predict(self, data):
        with torch.no_grad():
            node_pred, edge_pred = self(data)
        node_pred, edge_pred = torch.exp(node_pred)[:, 1], torch.exp(edge_pred)[:, 1]
            
        if hasattr(data,'edge_type'):
            node_mask = data.node_type == 0
            edge_mask = data.edge_type == 0
            node_pred = node_pred[node_mask]
            edge_pred = edge_pred[edge_mask]
            
        return node_pred, edge_pred

    def predict_nodes(self, data):
        node_pred, edge_pred = self.predict(data)
        return node_pred

    def predict_edges(self, data):
        node_pred, edge_pred = self.predict(data)
        return edge_pred
    

    def shared_step(self, batch, batch_idx, tag=None):
        node_o, edge_o = self(batch)
        
        if hasattr(batch,'edge_type'):
            node_mask = batch.node_type == 0
            edge_mask = batch.edge_type == 0
            
            node_o = node_o[node_mask]
            edge_o = edge_o[edge_mask]
            batch.y = batch.y[node_mask]
            batch.edge_y = batch.edge_y[edge_mask]
            
        # selected_edges = losses.select_top_edges(edge_o,batch)
        # t4eff = batch.edge_y[selected_edges].sum()/batch.edge_y.sum()

        loss = self.loss(self, node_o, edge_o, batch)
        
        node_acc = accuracy(node_o.argmax(dim=1),batch.y)
        edge_acc = accuracy(edge_o.argmax(dim=1),batch.edge_y)
        
        node_score = torch.exp(node_o[:,1])
        edge_score = torch.exp(edge_o[:,1])
        
        node_auroc = auroc(node_score, batch.y)
        edge_auroc = auroc(edge_score, batch.edge_y)

        true_node_score = node_score[batch.y==1]
        fake_node_score = node_score[batch.y==0]
        true_edge_score = edge_score[batch.edge_y==1]
        fake_edge_score = edge_score[batch.edge_y==0]
        
        metrics = dict(
            loss=loss,
            node_acc=node_acc,
            edge_acc=edge_acc,
            node_auroc=node_auroc, edge_auroc=edge_auroc
        )
        histos = dict(true_node_score=true_node_score, fake_node_score=fake_node_score, 
                      true_edge_score=true_edge_score, fake_edge_score=fake_edge_score)

        self.log_scalar(metrics, tag=tag)
        self.log_histo(histos, tag=tag)
        
        return metrics

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, tag='train')

    def validation_step(self, batch, batch_idx):
        metrics = self.shared_step(batch, batch_idx, tag='val')
        
        self.log('hp_metric', metrics['edge_auroc'])
        for key,value in metrics.items(): self.log(f'hp/{key}', value)
        
        return { f'val_{key}':value for key,value in metrics.items() }

    def test_step(self, batch, batch_idx):
        metrics = self.shared_step(batch, batch_idx, tag='test')
        return { f'test_{key}':value for key,value in metrics.items() }

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        return [optimizer], dict(
            scheduler=scheduler,
            monitor='hp/loss'
        )
        
    def log_scalar(self, metrics, tag=None):
        for key,scalar in metrics.items():
            self.log(f'{key}/{tag}',scalar)
        
    def log_histo(self, histos, tag=None):
        for key,histo in histos.items():
            self.logger.experiment.add_histogram(
                f'{key}/{tag}',
                histo,
                self.current_epoch,
            )