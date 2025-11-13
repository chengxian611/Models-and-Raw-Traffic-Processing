from .transformer import Encoder, FFN
from .graphsage import GraphSAGE
import torch.nn as nn
from .base import BaseModel
from ..config import Config
import torch
class FTASGNN(BaseModel):

    def __init__(self, input_dim, hidden_dim, out_channels, seq_len=8, heads=1, dropout=0.1, aggr='mean', latent_dim_scale=2):
        super().__init__()
        self.name = "TA-PGCN"
        self.encoder = Encoder(heads=heads, input_dim=input_dim, seq_len=seq_len, 
                                       d_model=hidden_dim, latent_dim_scale=latent_dim_scale)
        self.graphsage = GraphSAGE(in_channels=hidden_dim, hidden_channels=hidden_dim, out_channels=hidden_dim, aggr=aggr)
        self.ffn = FFN(d_model=hidden_dim, out_channels=out_channels, dropout=dropout, latent_dim_scale=latent_dim_scale)
        self.f1 = nn.Linear(in_features=hidden_dim, out_features=out_channels)
        # self.encoder.register_forward_hook(self._register_hook('after_encoder'))
        self.graphsage.register_forward_hook(self._register_hook('after_graphsage'))
        self.time_embedding = nn.Sequential(
            nn.BatchNorm1d(1),
            nn.Linear(1, hidden_dim),
            # nn.LayerNorm(seq_len*hidden_dim)
        )
    def forward(self, blocks, x):
        # x -> batch_size, seq_len, input_dim
        if x.dim() == 2:
            x = x.reshape(x.shape[0], Config.seq_len, Config.d_packet)
        packet_time_stamp = None
        
        if "packet_time_stamps" in blocks[0].ndata:
            # Extract the actual Tensor from the dictionary
            # Assuming the default node type for a homogeneous graph is blocks[0].ntypes[0]
            node_type = blocks[0].ntypes[0] if blocks[0].ntypes else '_N'
            if node_type in blocks[0].ndata["packet_time_stamps"]:
                packet_time_stamp = blocks[0].ndata["packet_time_stamps"][node_type]
        # x -> batch_size, seq_len, input_dim
        x= self.encoder(x, packet_time_stamp)
        
        # x -> batch_size, seq_len, d_model 
        x = x[:, 0, :]
        # x -> batch_size, d_model
        session_time_stamp = None
        if "session_time_stamps" in  blocks[0].ndata:
            node_type = blocks[0].ntypes[0] if blocks[0].ntypes else '_N'
            if node_type in blocks[0].ndata["session_time_stamps"]:
                session_time_stamp = blocks[0].ndata["session_time_stamps"][node_type]
        t = self.time_embedding(session_time_stamp) if session_time_stamp is not None \
            else torch.zeros(x.shape[0], Config.HIDDEN_DIM, device=x.device)
        x = x + t  
        x = self.graphsage(blocks, x)
        x = self.ffn(x)
        return x
