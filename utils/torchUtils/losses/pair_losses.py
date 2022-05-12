
import torch.nn.functional as F
from torch import Tensor
import torch

def std_loss(model, pair_o, batch, **kwargs):
    loss = F.nll_loss(pair_o, batch.pair_y)
    return loss

mismatch_weights = Tensor([4.2755, 1.1581, 1.0000, 7.2572])
def mismatched_bjet_loss(model, pair_o, batch, **kwargs):
    weights = mismatch_weights.to(model.device)

    ni, nj = batch.pair_index
    ni_y, nj_y = batch.y[ni], batch.y[nj]

    n_same = ni_y == nj_y
    n_diff = ni_y ^  nj_y

    def _get_loss(mask):
        if mask.sum() == 0: return 0
        return F.nll_loss(pair_o[mask], batch.pair_y[mask])

    def _get_weighted_loss(mask, weights):
        if mask.sum() == 0: return 0
        return F.nll_loss(pair_o[mask], batch.pair_y[mask], weights)

    loss_00 = weights[0]*_get_loss((ni_y == 0) & n_same)
    loss_01 = weights[1]*_get_loss(n_diff)
    loss_11 = _get_weighted_loss((ni_y == 1) & n_same, weights[2:])
    return model.type_weights[1] * (loss_00 + loss_01 + loss_11)