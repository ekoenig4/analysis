import torch
from torch import nn
from collections import defaultdict

from .torch_tools import aggregate_products
from .Feynman import Feynman

def get_activation(act):
    return dict(
        relu=nn.ReLU,
        gelu=nn.GELU,
        leakyrelu=nn.LeakyReLU,
    ).get(act)

class FeatureConv(nn.Module):
    """FeatureConv used in ParticleNet 
       https://github.com/ekoenig4/weaver-multiH/blob/62028451d7b3f06de7abc3d83dfd7bdfb1b2d403/weaver/utils/nn/model/ParticleNet.py#L220
    """
    def __init__(self, channels=[], act='leakyrelu', **kwargs):
        super(FeatureConv, self).__init__(**kwargs)

        assert len(channels) > 1, 'Need at least 2 channels to convolve'

        act = get_activation(act)

        self.conv = nn.Sequential(
            nn.BatchNorm1d(channels[0]),
            *[
                nn.Conv1d(n_in, n_out, kernel_size=1, bias=False)
                for n_in, n_out in zip(channels[:-1], channels[1:])
            ],
            nn.BatchNorm1d(channels[-1]),
            act(),
            )

    def forward(self, x):
        return self.conv(x)

class FeynNet(nn.Module):
    def __init__(self,
                 diagram: Feynman,
                 nfinalstates : dict = {},
                 particle_aggr='max',
                 particle_conv = [32,32],
                 particle_act='leakyrelu',
                 **kwargs
                 ):
        """Feynman Network for permutation invariant final state reconstruction

        Args:
            diagram (Feynman): Feynman object diagram
            nfinalstates (dict, optional): multiplicity for each finalstate type. Defaults to {}.
            particle_aggr (str, optional): aggregate function for product features. Defaults to 'max'.
            particle_conv (list or dict, optional): layer parameters for FeatureConv. If list given, will use the same parameters for all particles. 
                Different parameters can be defined for sepearate particles using a dictionary instead. Defaults to [32,32].
        """
        super(FeynNet, self).__init__(**kwargs)

        self.diagram = diagram.build_diagram()

        # get internalstate particles
        internalstates = self.diagram.get_internalstate_types()

        # get product permutations for each internal particle to be reconstructed
        product_assignments = {
            particle_type: particles[0].get_product_permutations(**nfinalstates)
            for particle_type, particles in internalstates.items()
        }

        # store assignment tensors to buffer so they can be passed to gpu
        self.particle_products = defaultdict(dict)
        for particle_type, products in product_assignments.items():
            for product_type, assignment in products.items():
                self.particle_products[particle_type][product_type] = f'{particle_type}_{product_type}_assignment'
                self.register_buffer(f'{particle_type}_{product_type}_assignment', torch.from_numpy(assignment) )

        # TODO: allow different aggrs for different particles?
        self.particle_aggr = particle_aggr

        # if list given, use same conv params for all particles
        if isinstance(particle_conv, list):
            particle_conv = {
                particle_type:particle_conv
                for particle_type in internalstates
            }

        # setup feature convolutions for each particle
        self.particle_mlps = nn.ModuleDict({
            particle_type:nn.Sequential(*[
                FeatureConv((n_in, n_out), act=particle_act)
                for n_in, n_out in zip(particle_conv[particle_type][:-1], particle_conv[particle_type][1:])
            ])
            for particle_type in internalstates
        })
        
    def get_product_assignment(self, particle_type):
        products = self.particle_products[particle_type]
        product_assignment = { product:getattr(self, assignment) for product, assignment in products.items() }
        return product_assignment
    
    def aggregate_products(self, particle_type, **features):
        product_assignment = self.get_product_assignment(particle_type)
        return aggregate_products(product_assignment, aggr=self.particle_aggr, **features)

    def forward(self, return_features=False, **features):
        
        for particle_type in self.particle_products:
            particle_features = self.aggregate_products(particle_type, **features)
            features[particle_type] = self.particle_mlps[particle_type](particle_features)

        if return_features:
            return features

        return features[particle_type]

    def reduce(self, reduction, **features):

        for particle_type, products in self.particle_products.items():
            product_assignment = { product:getattr(self, assignment) for product, assignment in products.items() }
            particle_features = aggregate_products(product_assignment, aggr=self.particle_aggr, **features)
            features[particle_type] = reduction(particle_features)

        return features
