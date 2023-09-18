from . import *

gen_info = dict(
    pt=lambda n : np.random.exponential(50, size=n),
    eta=lambda n : np.random.normal(0, 1, size=n),
    phi=lambda n : np.random.uniform(-np.pi, np.pi, size=n),
    m=lambda n : np.random.exponential(400, size=n)+400,
)

def as_p4(array):
    return ak.zip(dict(
        pt = array.pt,
        eta = array.eta,
        phi = array.phi,
        m = array.m,
    ), with_name='Momentum4D')

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

class X_ttH_8j(Generator):
    def valid(self, event):
        hm = np.random.normal(125, 10, size=(len(event)))
        tm = np.random.normal(172, 10, size=(len(event), 2))

        mask = event.m > ( hm + np.sum(tm, axis=1) )

        self.hm = hm[mask]
        self.tm = tm[mask]

        return mask

    def physics(self, event):
        t1, t2, h1 = three_body_decay(event, self.tm[:,0], self.tm[:,1], self.hm)
        t1, t2 = swap_objects(t1, t2, t1.pt < t2.pt)

        wm = np.random.normal(80, 10, size=(2, len(event)))
        bm = np.random.lognormal(np.log(5), 0.5, size=(4, len(event)))
        jm = np.random.lognormal(np.log(5e-3), 0.5, size=(4, len(event)))

        t1_w, t1_b = two_body_decay(t1, wm[0], bm[0])
        t1_w_j1, t1_w_j2 = two_body_decay(t1_w, jm[0], jm[1])
        t1_w_j1, t1_w_j2 = swap_objects(t1_w_j1, t1_w_j2, t1_w_j1.pt < t1_w_j2.pt)

        t2_w, t2_b = two_body_decay(t2, wm[1], bm[1])
        t2_w_j1, t2_w_j2 = two_body_decay(t2_w, jm[2], jm[3])
        t2_w_j1, t2_w_j2 = swap_objects(t2_w_j1, t2_w_j2, t2_w_j1.pt < t2_w_j2.pt)

        h1_b1, h1_b2 = two_body_decay(h1, bm[2], bm[3])
        h1_b1, h1_b2 = swap_objects(h1_b1, h1_b2, h1_b1.pt < h1_b2.pt)

        quarks = [t1_w_j1, t1_w_j2, t1_b, t2_w_j1, t2_w_j2, t2_b, h1_b1, h1_b2]
        quarks = [ as_p4(q) for q in quarks ]

        Q = ak.concatenate([ q[:,None] for q in quarks ], axis=1)
        hits = hadronization(Q, maxiters=30)
        jet = cluster_jets(hits)
        jet = jet[jet.pt > 10]
        jet['signalId'] = gen_match_jets(jet, quarks)
        
        return dict(
            t1 = t1,
            t2 = t2,
            h1 = h1,
            t1_w = t1_w,
            t1_b = t1_b,
            t1_w_j1 = t1_w_j1,
            t1_w_j2 = t1_w_j2,
            t2_w = t2_w,
            t2_b = t2_b,
            t2_w_j1 = t2_w_j1,
            t2_w_j2 = t2_w_j2,
            h1_b1 = h1_b1,
            h1_b2 = h1_b2,
            jet = jet
        )
    
class QCD(Generator):
    def physics(self, event):
        quarks = hadronization(event, maxiters=30 + 8)
        jet = cluster_jets(quarks)
        jet = jet[jet.pt > 10]
        jet['signalId'] = -ak.ones_like(jet.pt, dtype=int)

        return dict(
            jet = jet
        )