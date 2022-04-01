from torch_geometric.nn import Linear
import torch
import torch.nn.functional as F

from .cpp_geometric import layers
from .LightningModel import LightningModel

class GCN(LightningModel):
    output = 'models/graph_gcn'
    name = 'graph_gcn'
    def __init__(self, dataset, nn_conv1_out=32, nn_conv2_out=128, **kwargs):
        super().__init__(dataset, **kwargs)

        nn1 = torch.nn.Sequential(
            Linear(2*len(dataset.node_attr_names) +
                   len(dataset.edge_attr_names), nn_conv1_out),
            torch.nn.ELU()
        )

        self.conv1 = layers.EdgeConv(
            nn1, edge_aggr=None, return_with_edges=True)

        nn2 = torch.nn.Sequential(
            Linear(5*nn_conv1_out, nn_conv2_out),
            torch.nn.ELU()
        )

        self.conv2 = layers.EdgeConv(
            nn2, edge_aggr=None, return_with_edges=True)

        self.edge_seq = torch.nn.Sequential(
            Linear(3*nn_conv2_out, 2),
        )

        self.node_seq = torch.nn.Sequential(
            Linear(nn_conv2_out, 2),
        )

    def forward(self, data):
        if type(data) is list:
            data = data[0]
        if type(data) is not tuple:
            data = (data.x, data.edge_index, data.edge_attr)
        x, edge_index, edge_attr = data

        x, edge_attr = self.conv1(x, edge_index, edge_attr)
        x, edge_attr = self.conv2(x, edge_index, edge_attr)
        x, edge_attr = self.node_seq(x), self.edge_seq(edge_attr)

        return F.log_softmax(x, dim=1), F.log_softmax(edge_attr, dim=1)


class GoldenGCN(LightningModel):
    output = 'models/golden_classifier/'
    name = 'golden'
    def __init__(self, dataset=None, nn_conv1_out=32, nn_conv2_out=128, nn_node_out=64, nn_edge_out=64, node_attr_names=None, edge_attr_names=None, **kwargs):
        super().__init__(dataset, node_attr_names=node_attr_names, edge_attr_names=edge_attr_names, **kwargs)
        self.save_hyperparameters('nn_conv1_out', 'nn_conv2_out', 'nn_node_out', 'nn_edge_out')
        
        if dataset is not None: 
            node_attr_names = dataset.node_attr_names
            edge_attr_names = dataset.edge_attr_names

        n_in_node = len(node_attr_names)
        n_in_edge = len(edge_attr_names)
        
        self.conv1 = layers.GCNConvMSG(n_in_node=n_in_node, n_in_edge=n_in_edge, n_out=nn_conv1_out)
        self.relu1 = layers.GCNRelu()
        self.conv2 = layers.GCNConvMSG(
            n_in_node=nn_conv1_out, n_in_edge=3*nn_conv1_out, n_out=nn_conv2_out)
        self.relu2 = layers.GCNRelu()

        self.node_linear1 = layers.NodeLinear(nn_conv2_out, nn_node_out)
        self.edge_linear1 = layers.EdgeLinear(3*nn_conv2_out, nn_edge_out)
        
        self.relu3 = layers.GCNRelu()
        
        self.node_linear2 = layers.NodeLinear(nn_node_out, 2)
        self.edge_linear2 = layers.EdgeLinear(nn_edge_out, 2)
        
        self.log_softmax = layers.GCNLogSoftmax()
        
        self.return_node_only = False

    def forward(self, data=None, x=None, edge_index=None, edge_attr=None):
        if data is None:
            data = (x,edge_index,edge_attr)
        if type(data) is list:
            data = data[0]
        if type(data) is not tuple:
            data = (data.x, data.edge_index, data.edge_attr)
        x, edge_index, edge_attr = data

        x, edge_attr = self.conv1(x, edge_index, edge_attr)
        x, edge_attr = self.relu1(x, edge_index, edge_attr)

        x, edge_attr = self.conv2(x, edge_index, edge_attr)
        x, edge_attr = self.relu2(x, edge_index, edge_attr)

        x, edge_attr = self.node_linear1(x, edge_index, edge_attr)
        x, edge_attr = self.edge_linear1(x, edge_index, edge_attr)
        x, edge_attr = self.relu3(x, edge_index, edge_attr)

        x, edge_attr = self.node_linear2(x, edge_index, edge_attr)
        x, edge_attr = self.edge_linear2(x, edge_index, edge_attr)
        x, edge_attr = self.log_softmax(x, edge_index, edge_attr)

        if self.return_node_only: return x
        return x, edge_attr
    
class GoldenGCN_Trim(LightningModel):
    output = 'models/golden_trim/'
    name = 'golden_trim'
    def __init__(self, dataset, nn_conv1_out=32, nn_conv2_out=128, **kwargs):
        super().__init__(dataset, **kwargs)
        self.save_hyperparameters('nn_conv1_out', 'nn_conv2_out')

        n_in_node = len(dataset.node_attr_names)
        n_in_edge = len(dataset.edge_attr_names)
        
        self.conv1 = layers.GCNConvMSG(n_in_node=n_in_node, n_in_edge=n_in_edge, n_out=nn_conv1_out)
        self.trim1 = layers.TrimEdges(n_in=3*nn_conv1_out)
        self.relu1 = layers.GCNRelu()
        self.conv2 = layers.GCNConvMSG(
            n_in_node=nn_conv1_out, n_in_edge=3*nn_conv1_out+1, n_out=nn_conv2_out)
        self.relu2 = layers.GCNRelu()

        self.node_linear = layers.NodeLinear(nn_conv2_out, 2)
        self.edge_linear = layers.EdgeLinear(3*nn_conv2_out, 2)
        self.log_softmax = layers.GCNLogSoftmax()

    def forward(self, data):
        if type(data) is list:
            data = data[0]
        if type(data) is not tuple:
            data = (data.x, data.edge_index, data.edge_attr)
        x, edge_index, edge_attr = data

        x, edge_attr = self.conv1(x, edge_index, edge_attr)
        edge_index,edge_attr = self.trim1(x, edge_index, edge_attr)
        
        x, edge_attr = self.relu1(x, edge_index, edge_attr)

        x, edge_attr = self.conv2(x, edge_index, edge_attr)
        x, edge_attr = self.relu2(x, edge_index, edge_attr)

        x, edge_attr = self.node_linear(x, edge_index, edge_attr)
        x, edge_attr = self.edge_linear(x, edge_index, edge_attr)

        x, edge_attr = self.log_softmax(x, edge_index, edge_attr)
        
        # -- Fix missing edges, set them to 0
        edge_mask = self.trim1.edge_mask
        edge_o = torch.zeros(2*edge_mask.shape[0]).reshape(-1,2).to("cuda:0")
        edge_o[:,0] = 100
        edge_o[:,1] = -100
        edge_o[edge_mask] = edge_attr
        
        return x, edge_o

modelMap = { value.name:value for value in locals().values() if isinstance(value,type) and issubclass(value,LightningModel) and hasattr(value,'name')}