import torch 

def group_products(product_assignment : dict, **features):
    assert isinstance(product_assignment, dict), 'unrecognized product assignment'

    grouped_features = torch.concat([ features[particle_type][:, :, permutations] for particle_type, permutations in product_assignment.items() ], axis=-1)
    return grouped_features

def aggregate_products(product_assignment : dict, aggr='max', **features):
    assert aggr in ['cat','sum','avg','max'], f'unrecongnized aggr {aggr}. expected cat, sum, avg, or max'

    grouped_features = group_products(product_assignment, **features)

    if aggr == 'cat': return grouped_features
    if aggr == 'sum': return grouped_features.sum(axis=3)
    if aggr == 'avg': return grouped_features.mean(axis=3)
    if aggr == 'max': return grouped_features.max(axis=3)[0]