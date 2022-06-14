from torch.utils.data import Sampler
from torch_geometric.data import Data

class BaseFilter(Sampler):
    def __init__(self, dataset, **kwargs):
        self.dataset = dataset        
        for key,value in kwargs.items(): 
            setattr(self, key, value)
        self.slice = [ i for i,data in enumerate(self.dataset) if self(data) ]

    def __iter__(self): return iter(self.slice)
    def __len__(self): return len(self.slice)

class Require8B(BaseFilter):
    def __call__(self, data : Data) -> bool:
        return data.y.sum() == 8
        