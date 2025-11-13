"""GraphSAGE model implementation."""
from ..config import Config
import torch.nn as nn
import dgl.function as fn
import dgl.nn.pytorch as dglnn
from .base import BaseModel
import torch.nn.functional as F

class GraphSAGE(BaseModel):
    """GraphSAGE model implementation."""
    
    def __init__(self, in_channels, hidden_channels, out_channels, aggr='mean'):
        super(GraphSAGE, self).__init__()
        self.name = "GraphSAGE"
        self.conv1 = dglnn.SAGEConv(in_channels, hidden_channels, aggregator_type=aggr)
        self.conv2 = dglnn.SAGEConv(hidden_channels, out_channels, aggregator_type=aggr)
        self.conv = dglnn.SAGEConv(in_channels, out_channels, aggregator_type=aggr)
        self.dropout = nn.Dropout(0.5)
        self.conv2.register_forward_hook(self._register_hook('after_graphsage'))
    def forward(self, blocks, x):
        if "TA" not in Config.MODEL_NAMES[Config.MODELS_SELECT]:
            x = x.reshape(x.shape[0], -1)
            x = x[:,:Config.d_session]
        x = self.conv1(blocks[0], x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(blocks[1], x)
        return x 
