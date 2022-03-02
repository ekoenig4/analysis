from ..utils import *

def get_remaining_pairs(pair,all_pairs):
    return list(filter(lambda h : not any(j in h for j in pair),all_pairs))

def get_quadH_pairings(njets=8,pair_index=True):
    if njets < 8: return
    jets = list(range(njets))
    all_pairs = [ (j0,j1) for j0 in jets for j1 in jets[j0+1:] ]
    quad_h = []
    for i,h0 in enumerate(all_pairs):
        second_pairs = get_remaining_pairs(h0,all_pairs[i+1:])
        for j,h1 in enumerate(second_pairs):
            third_pairs = get_remaining_pairs(h1,second_pairs[j+1:])
            for k,h2 in enumerate(third_pairs):
                fourth_pairs = get_remaining_pairs(h2,third_pairs[k+1:])
                for h3 in fourth_pairs:
                    pair = [ all_pairs.index(h) for h in (h0,h1,h2,h3) ] if pair_index else [ j for h in (h0,h1,h2,h3) for j in h ]
                    quad_h.append(pair)
    return quad_h

quadh_index = get_quadH_pairings()

def mass_diff(dijets):
    ms = dijets.m[:,quadh_index]
    ms_numpy = ms.to_numpy().reshape(-1,4)
    maxim,minim = np.max(ms_numpy,axis=-1),np.min(ms_numpy,axis=-1)
    m_asym = (maxim-minim)/(maxim+minim)
    return m_asym.reshape(-1,105)

def pair_higgs(tree,operator=mass_diff,nsave=4):
    dijets = get_collection(tree,'dijet',False)
    score = ak.from_regular(operator(dijets),axis=-1)
    best_score = ak.argsort(score,axis=-1)
    score = score[best_score]
    form = {0:''}
    for save in range(nsave):
        best_index = ak.from_regular(best_score[:,save])
        dijet_index = ak.from_regular(np.array(quadh_index)[best_index],axis=-1)
        higgs = dijets[dijet_index]
        higgs = higgs[ak.argsort(-higgs.pt,axis=-1)]
        tree.extend(
            rename_collection(higgs,f'higgs{form.get(save,save)}'),
            **{
                f'score_4h{form.get(save,save)}':score[:,save],
                f'nfound_h{form.get(save,save)}':ak.sum(higgs.signalId>-1,axis=-1),
                f'h{form.get(save,save)}_msum':ak.sum(higgs.m,axis=-1),
                f'h{form.get(save,save)}_ptsum':ak.sum(higgs.pt,axis=-1)
                })
    return tree