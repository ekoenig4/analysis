from . import *

gen_info = dict(
    pt = lambda n : np.random.lognormal( np.log(50) , 1, size=n),
    eta= lambda n : np.where(np.random.choice([0, 1], size=n),
                            np.random.normal(-2.85, 1.5, size=n),
                            np.random.normal( 2.85, 1.5, size=n),
    ),
    phi= lambda n : np.random.uniform(-np.pi, np.pi, size=n),
    m =  lambda n : np.random.lognormal( np.log(400) , 1, size=n),
)

def swap_objects(obj1, obj2, mask):
    new_obj1 = ak.zip(dict(
        pt = ak.where(mask, obj2.pt, obj1.pt),
        eta = ak.where(mask, obj2.eta, obj1.eta),
        phi = ak.where(mask, obj2.phi, obj1.phi),
        m = ak.where(mask, obj2.m, obj1.m),
    ), with_name='Momentum4D')

    new_obj2 = ak.zip(dict(
        pt = ak.where(mask, obj1.pt, obj2.pt),
        eta = ak.where(mask, obj1.eta, obj2.eta),
        phi = ak.where(mask, obj1.phi, obj2.phi),
        m = ak.where(mask, obj1.m, obj2.m),
    ), with_name='Momentum4D')

    return new_obj1, new_obj2

class X_HH_4b(Generator):
    def valid(self, event):
        hm = np.random.normal(125, 10, size=(len(event), 2))
        mask = event.m > np.sum(hm, axis=1)
        self.hm = hm[mask]

        return mask

    def physics(self, event):
        h1, h2 = two_body_decay(event, self.hm[:,0], self.hm[:,1])

        h1, h2 = swap_objects(h1, h2, h1.pt < h2.pt)

        bm = np.random.lognormal(np.log(5), 0.5, size=(4, len(event)))

        h1b1, h1b2 = two_body_decay(h1, bm[0], bm[1])
        h1b1, h1b2 = swap_objects(h1b1, h1b2, h1b1.pt < h1b2.pt)

        h2b1, h2b2 = two_body_decay(h2, bm[2], bm[3])
        h2b1, h2b2 = swap_objects(h2b1, h2b2, h2b1.pt < h2b2.pt)

        quarks = ak.concatenate([ b[:,None] for b in (h1b1, h1b2, h2b1, h2b2)], axis=1)
        hits = hadronization(quarks, maxiters=30)
        jet = cluster_jets(hits)
        jet = jet[jet.pt > 10]
        jet['signalId'] = gen_match_jets(jet, (h1b1, h1b2, h2b1, h2b2))
        
        return dict(
            h1 = h1,
            h2 = h2,
            h1b1 = h1b1,
            h1b2 = h1b2,
            h2b1 = h2b1,
            h2b2 = h2b2,
            jet = jet
        )
    
class QCD(Generator):
    def physics(self, event):
        quarks = hadronization(event, maxiters=30 + 4)
        jet = cluster_jets(quarks)
        jet = jet[jet.pt > 10]
        jet['signalId'] = -ak.ones_like(jet.pt, dtype=int)

        return dict(
            jet = jet
        )