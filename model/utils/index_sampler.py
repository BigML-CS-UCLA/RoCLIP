
from torch.utils.data import Sampler
class IndexSampler(Sampler[int]):
    """
    Samples from the specified indices
    """
    def __init__(
        self, 
        indices
    ):
        self.indices = indices 

    def __iter__(self):
        return iter(self.indices)
    
    def __len__(self):
        return len(self.indices)