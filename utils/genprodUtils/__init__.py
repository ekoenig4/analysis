import numpy as np
import awkward as ak


def chunk_array(array, nbatches=4):
    indicies = np.arange(len(array))
    for batch in np.array_split(indicies, nbatches):
        yield array[batch]

from .algorithms import cluster_jets, gen_match_jets    
from .rng import norm_exponential, ak_rand_like, random_vector_like
from .kinematics import radiate, two_body_decay, hadronization, three_body_decay 