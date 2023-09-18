import awkward as ak
import numpy as np
from tqdm import tqdm

from .rng import ak_rand_like, random_vector_like

def radiate(p4, energy):
    rvec = random_vector_like(p4)
    rvec = rvec.add(p4.to_Vector3D())
    rvec = (rvec / np.sqrt( (rvec.x**2 + rvec.y**2 + rvec.z**2) ))
    
    r_p4  = ak.zip(dict(
        px=energy * rvec.x,
        py=energy * rvec.y,
        pz=energy * rvec.z,
        e=energy,
    ), with_name='Momentum4D')
    
    p4 = ak.zip(dict(
        px=p4.px - r_p4.px,
        py=p4.py - r_p4.py,
        pz=p4.pz - r_p4.pz,
        m=p4.m,
    ), with_name='Momentum4D')
    
    return p4, r_p4
    
def two_body_decay(x, m1, m2):
    
    rvec = random_vector_like(x)
    
    if ak.all(x.p == 0):
        x_rest = x
    else:
        x_rest = x.boost_p4(-x)
    
    if isinstance(m1, (int, float)) or callable(m1):
        m1 = m1(len(x)) if callable(m1) else np.ones( len(x) ) * m1
        if x.ndim == 2:
            m1 = m1.reshape(-1, 1)
        
    if isinstance(m2, (int, float)) or callable(m2):
        m2 = m2(len(x)) if callable(m2) else np.ones( len(x) ) * m2
        if x.ndim == 2:
            m2 = m2.reshape(-1, 1)
        
        
    e1 = (m1**2 + x_rest.mass**2 - m2**2) / (2 * x_rest.mass)
    p1 = np.sqrt(e1**2 - m1**2)
    p1 = ak.zip(dict(
        px=p1 * rvec.x,
        py=p1 * rvec.y,
        pz=p1 * rvec.z,
        m=m1,
    ), with_name='Momentum4D')
    
    e2 = (m2**2 + x_rest.mass**2 - m1**2 ) / (2 * x_rest.mass)
    p2 = np.sqrt(e2**2 - m2**2)
    p2 = ak.zip(dict(
        px=-p2 * rvec.x,
        py=-p2 * rvec.y,
        pz=-p2 * rvec.z,
        m=m2,
    ), with_name='Momentum4D')
    
    p1 = p1.boost_p4(x)
    p2 = p2.boost_p4(x)
    
    return p1, p2

def three_body_decay(x, m1, m2, m3):
    rvec = random_vector_like(x)
    
    if ak.all(x.p == 0):
        x_rest = x
    else:
        x_rest = x.boost_p4(-x)
    
    if isinstance(m1, (int, float)) or callable(m1):
        m1 = m1(len(x)) if callable(m1) else np.ones( len(x) ) * m1
        if x.ndim == 2:
            m1 = m1.reshape(-1, 1)
        
    if isinstance(m2, (int, float)) or callable(m2):
        m2 = m2(len(x)) if callable(m2) else np.ones( len(x) ) * m2
        if x.ndim == 2:
            m2 = m2.reshape(-1, 1)

    if isinstance(m3, (int, float)) or callable(m3):
        m3 = m3(len(x)) if callable(m3) else np.ones( len(x) ) * m3
        if x.ndim == 2:
            m3 = m3.reshape(-1, 1)

    M = ak.concatenate([ m1[:,None], m2[:,None], m3[:,None] ], axis=1)

    s = ak_rand_like(lambda n : np.random.uniform(size=n), M)
    randarg = ak.argsort(s, axis=-1)
    undoarg = ak.argsort(randarg, axis=-1)

    M = M[randarg]
    m1, m2, m3 = M[:,0], M[:,1], M[:,2]

    max_e1 = (m1**2 + x_rest.mass**2 - (m2 + m3)**2) / (2 * x_rest.mass)

    e1 = np.random.uniform(size=len(x)) * (max_e1 - m1) + m1
    p1 = np.sqrt(e1**2 - m1**2)

    p1 = ak.zip(dict(
        px = p1 * rvec.x,
        py = p1 * rvec.y,
        pz = p1 * rvec.z,
        m=m1,
    ), with_name='Momentum4D')
    
    remaining = x_rest - p1
    p2, p3 = two_body_decay(remaining, m2, m3)
    
    p1 = p1.boost_p4(x)
    p2 = p2.boost_p4(x)
    p3 = p3.boost_p4(x)

    P = ak.concatenate([ p1[:,None], p2[:,None], p3[:,None] ], axis=1)
    P = P[undoarg]

    return P[:,0], P[:,1], P[:,2]

# def hadronization(p4, maxiters=10, loss=0.95, maxE=1, threshold=0.9, bar=None, it=0):
#     if bar is None:
#         bar = tqdm(total=maxiters)        
        
#     if p4.ndim == 1:
#         p4 = p4[:,None]
    
#     queue = [ (0,p4) ]
#     hits = []
    
    
#     while len(queue) > 0:
#         it, p4 = queue.pop(0)
        
#         if it >= maxiters:
#             hits.append(p4)
#             continue
        
#         if np.mean(p4.e < maxE) > threshold:
#             hits.append(p4)
#             continue        
        
#         r1 = ak_rand_like( lambda n : np.random.uniform(0.01, 0.9, n), p4)
#         r2 = ak_rand_like( lambda n : np.random.uniform(0.01, 1, n), p4) * (loss - r1)
        
#         m1, m2 = p4.m * r1, p4.m * r2
#         parts = ak.concatenate(two_body_decay(p4, m1, m2), axis=1)
#         queue.append( (it+1, parts) )
#         bar.update(1)
        
#     return ak.concatenate(hits, axis=1)

def random_decay(p4, decay_prob=0.25, maxE=0.1):
    to_decay = ak_rand_like( lambda n : np.random.uniform(0, 1, n), p4) < decay_prob
    to_decay = to_decay & (p4.E > maxE)

    decaying = p4[to_decay]
    propagtting = p4[~to_decay]

    loss = ak_rand_like( lambda n : np.random.uniform(size=n), decaying)
    r1 = loss * ak_rand_like( lambda n : np.random.uniform(size=n), decaying)
    r2 = loss - r1

    m1, m2 = decaying.m * r1, decaying.m * r2
    decayed = ak.concatenate(two_body_decay(decaying, m1, m2), axis=1)

    return ak.zip(dict(
        pt = ak.concatenate([propagtting.pt, decayed.pt], axis=1),
        eta = ak.concatenate([propagtting.eta, decayed.eta], axis=1),
        phi = ak.concatenate([propagtting.phi, decayed.phi], axis=1),
        m = ak.concatenate([propagtting.mass, decayed.mass], axis=1),
    ), with_name='Momentum4D')
    # return ak.concatenate([propagtting, decayed], axis=1)

def hadronization(p4, maxiters=10, decay_prob=0.25, maxE=0.1):
    if p4.ndim == 1: p4 = p4[:,None]

    parts = None
    for _ in tqdm(range(maxiters), desc='Hadronization'):
        if parts is None: parts = p4
        parts = random_decay(parts, decay_prob, maxE)

    return parts
