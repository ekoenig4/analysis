
import pytorch_lightning as pl
import torch
from ..gnn import to_tensor

class LightningModel(pl.LightningModule):
    def __init__(self, dataset=None, lr=1e-3, node_attr_names=None, edge_attr_names=None, scale=None, **hparams):
        super().__init__()
        self.lr = lr
        
        if dataset is not None:
            scale = dataset.scale 
            node_attr_names = dataset.node_attr_names
            edge_attr_names = dataset.edge_attr_names
            
        self.hparams['lr'] = lr            
        self.hparams['scale'] = scale
        self.hparams['node_attr_names'] = node_attr_names
        self.hparams['edge_attr_names'] = edge_attr_names

        for hparam, value in hparams.items():
            self.hparams[hparam] = value

        self.n_in_node = len(node_attr_names)
        self.n_in_edge = len(edge_attr_names)
        
        if dataset is not None:
            self.node_weights = to_tensor(dataset.node_class_weights)
            self.edge_weights = to_tensor(dataset.edge_class_weights)
            self.type_weights = to_tensor(dataset.type_class_weights)

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, tag='train')

    def validation_step(self, batch, batch_idx):
        metrics = self.shared_step(batch, batch_idx, tag='val')
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
        
    def log_histos(self, histos, tag=None):
        for key,histo in histos.items():
            self.logger.experiment.add_histogram(
                f'{key}/{tag}',
                histo,
                self.current_epoch,
            )