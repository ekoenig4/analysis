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