from torch_geometric.loader import DataLoader

def graph_loader(dataset, sampler=None, **kwargs):
    if sampler is not None: sampler = sampler(dataset)
    return DataLoader(dataset, **kwargs, sampler=sampler)
