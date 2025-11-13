import torch
import dgl
from dgl.dataloading import NeighborSampler

class MaskSampler(NeighborSampler):
    """
    Mask-aware neighbor sampler.
    Uses DGL's mask parameter for mask sampling.
    """
    def __init__(self, fanouts, mask=None):
        super().__init__(
            fanouts,
            mask=mask,  # Use edge labels as mask
            edge_dir='in',
            replace=False
        )
        self.mask = mask
    
    def sample(self, g, seed_nodes):
        if  self.mask not in g.edata:
            import warnings
            warnings.warn("Mask is set to None ")
        return self.sample_blocks(g, seed_nodes)

    def sample_blocks(self, g, seed_nodes, exclude_eids=None):
        return super().sample_blocks(g, seed_nodes, exclude_eids)
