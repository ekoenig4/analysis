from pyexpat import model

from .LightningModel import LightningModel
from .node_classifier import *
from .edge_classifier import *
from .cluster_classifier import *
from .hyper_edge_classifier import *
from .pair_classifier import *
from .simple_pair_classifier import *
from .quadH_classifier import *
from .attention_classifier import *

__all__ = ["modelMap"]

def debug(model):
    print(vars(model))
    return model


modelMap = {
    model.name : model
    for model in locals().values()
    if isinstance(model, type) and issubclass(model, LightningModel) and hasattr(model, 'name')
}