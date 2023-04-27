from multiprocessing import Pool
import itertools
from collections import defaultdict
import numpy as np

class SliceIterator:

    @classmethod
    def slice(cls, it, njobs):
        its = itertools.tee(it, njobs)
        return [ cls(it, njobs, i) for i, it in enumerate(its) ]

    def __init__(self, it, njobs, ijob):
        self.it = it
        self.njobs = njobs
        self.ijob = ijob
    def __iter__(self):
        for i, p in enumerate(self.it):
            if i%self.njobs == self.ijob:
                yield i, p

def _get_invariant_permutations(feynman, permiter, nfinalstate_types):
    invariant_reco_ids = []
    # invariant_reco_ids = defaultdict(lambda:0)
    finalstate_permutations = defaultdict(list)

    for i, permutations in permiter:
        permutations = {
            key: permutation[:nfinalstate]
            for key, nfinalstate, permutation in zip(nfinalstate_types.keys(), nfinalstate_types.values(), permutations)
        }
        reco_id = feynman.get_reco_id(**permutations)
        # invariant_reco_ids[reco_id] += 1
        if reco_id not in invariant_reco_ids:
        # if invariant_reco_ids[reco_id] == 1:
            invariant_reco_ids.append(reco_id)
            finalstate_permutations[reco_id].append((i,permutations))
            # for typeid, permutation in permutations.items():
                # finalstate_permutations[typeid].append((i,permutation))
    return finalstate_permutations

def get_invariant_permutations(feynman, permiter, nfinalstate_types, n_jobs=2):
    permiters = SliceIterator.slice(permiter, n_jobs)
    with Pool(n_jobs) as pool:
        worker_finalstate_permutations = pool.starmap(_get_invariant_permutations, 
                                                      zip(itertools.repeat(feynman), 
                                                          permiters, 
                                                          itertools.repeat(nfinalstate_types)))
    
    finalstate_permutations = defaultdict(list)
    for permMap in worker_finalstate_permutations:
        for reco_id, perm in permMap.items():
            finalstate_permutations[reco_id].append(perm)

    perms = [ min(perms,key=lambda kv:kv[0])[0]  for perms in finalstate_permutations.values() ]
    perms = sorted(perms, key=lambda kv:kv[0])

    finalstate_type_permutations = defaultdict(list)
    for permutation in perms:
        for typeid, perm in permutation[1].items():
            finalstate_type_permutations[typeid].append(perm)
            
    finalstate_permutations = { key: np.array(permutations) for key, permutations in finalstate_type_permutations.items() }
    return finalstate_permutations
