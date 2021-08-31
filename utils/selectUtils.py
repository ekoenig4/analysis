#!/usr/bin/env python
# coding: utf-8

from . import *

from skimage.measure import LineModelND, ransac
import scipy

ordinal = lambda n : "%d%s"%(n,{1:"st",2:"nd",3:"rd"}.get(n if n<20 else n%10,"th"))
array_min = lambda array,value : ak.min(ak.concatenate(ak.broadcast_arrays(value,array[:,np.newaxis]),axis=-1),axis=-1)

def get_jet_index_mask(jets,index):
    """ Generate jet mask for a list of indicies """
    if hasattr(jets,'ttree'): jets = jets["jet_pt"]
    
    jet_index = ak.local_index( jets )
    compare , _ = ak.broadcast_arrays( index[:,None],jet_index )
    inter = (jet_index == compare)
    return ak.sum(inter,axis=-1) == 1

def exclude_jets(input_mask,exclude_mask):
    return (input_mask == True) & (exclude_mask == False)

def get_top_njet_index(tree,jet_index,njets=6):
    index_array = ak.local_index(jet_index)
    firstN_array = index_array < njets if njets != -1 else index_array != -1
    top_njet_index = jet_index[firstN_array]
    bot_njet_index = jet_index[firstN_array == False]
    return top_njet_index,bot_njet_index

def sort_jet_index_simple(tree,varbranch,jets=None,method=max):
    """ Mask of the top njet jets in varbranch """
    
    if jets is None: jets = tree.all_jets_mask
    
    polarity = -1 if method is max else 1
    
    sorted_array = ak.local_index(jets,axis=-1)
    if varbranch is not None: sorted_array = np.argsort(polarity * varbranch)
        
    selected_sorted_array = jets[sorted_array]
    selected_array = sorted_array[selected_sorted_array]
    return selected_array

def sort_jet_index(tree,variable=None,jets=None,method=max):
    """ Mask of the top njet jets in variable """
        
    varbranch = tree[variable] if variable else None
    if variable == "jet_eta": varbranch = np.abs(varbranch)
    return sort_jet_index_simple(tree,varbranch,jets,method=method)

def count_sixb_index(jet_index,sixb_jet_mask):
    """ Number of signal b-jets in index list """
    
    nevts = ak.size(jet_index,axis=0)
    compare , _ = ak.broadcast_arrays( sixb_jet_mask[:,None], jet_index)
    inter = (jet_index == compare)
    count = ak.sum(ak.flatten(inter,axis=-1),axis=-1)
    return count

def count_sixb_mask(jet_mask,sixb_jet_mask):
    """ Number of signal b-jets in jet mask """
    
    inter = jet_mask & sixb_jet_mask
    return ak.sum(inter,axis=-1)

def get_jet_position(jet_index,jet_mask):
    """ Get index positions of jets in sorted jet index list"""
    position = ak.local_index(jet_index,axis=-1)
    jet_position = position[ jet_mask[jet_index] ]
    return jet_position

def calc_dphi(phi_1,phi_2):
    phi_2,phi_1 = ak.broadcast_arrays(phi_2[:,np.newaxis],phi_1)
    dphi = phi_2 - phi_1
    shift_over = ak.where(dphi > np.pi,-2*np.pi,0)
    shift_under= ak.where(dphi <= -np.pi,2*np.pi,0)
    return dphi + shift_over + shift_under

def calc_deta(eta_1,eta_2):
    eta_2,eta_1 = ak.broadcast_arrays(eta_2[:,np.newaxis],eta_1)
    return eta_2 - eta_1

def calc_dr(eta_1,phi_1,eta_2,phi_2):
    deta = calc_deta(eta_1,eta_2)
    dphi = calc_dphi(phi_1,phi_2)
    dr = np.sqrt( deta**2 + dphi**2 )
    return dr

def get_ext_dr(eta_1,phi_1,eta_2,phi_2):
    dr = calc_dr(eta_1,phi_1,eta_2,phi_2)
    dr_index = ak.local_index(dr,axis=-1)
    
    dr_index = dr_index[dr!=0]
    dr_reduced = dr[dr!=0]
    
    imin_dr = ak.argmin(dr_reduced,axis=-1,keepdims=True)
    min_dr = ak.flatten(dr_reduced[imin_dr],axis=-1)
    imin_dr = ak.flatten(dr_index[imin_dr],axis=-1)
    
    imax_dr = ak.argmax(dr_reduced,axis=-1,keepdims=True)
    max_dr = ak.flatten(dr_reduced[imax_dr],axis=-1)
    imax_dr = ak.flatten(dr_index[imax_dr],axis=-1)
    return dr,min_dr,imin_dr,max_dr,imax_dr

# --- Standard Preselection --- #
def std_preselection(tree,ptcut=20,etacut=2.5,btagcut=None,btagcut_invert=None,jetid=1,puid=1,njetcut=None,njetcut_invert=None,min_drcut=None,qglcut=None,
                     passthrough=False,exclude_events_mask=None,exclude_jet_mask=None,include_jet_mask=None,**kwargs):
    def jet_pu_mask(puid=puid):
        puid_mask = (1 << puid) == tree["jet_puid"] & ( 1 << puid )
        low_pt_pu_mask = (tree["jet_pt"] < 50) & puid_mask
        return (tree["jet_pt"] >= 50) | low_pt_pu_mask
    
    jet_mask = tree.all_jets_mask
    event_mask = tree.all_events_mask
    
    if include_jet_mask is not None: jet_mask = jet_mask & include_jet_mask 
    if exclude_jet_mask is not None: jet_mask = exclude_jets(jet_mask,exclude_jet_mask)
        
    if not passthrough:
        if ptcut: jet_mask = jet_mask & (tree["jet_pt"] > ptcut)
        if etacut: jet_mask = jet_mask & (np.abs(tree["jet_eta"]) < etacut)
        if btagcut: jet_mask = jet_mask & (tree["jet_btag"] > btagcut)
        if btagcut_invert: jet_mask = jet_mask & (tree["jet_btag"] <= btagcut_invert)
        if jetid: jet_mask = jet_mask & ((1 << jetid) == tree["jet_id"] & ( 1 << jetid ))
        if puid: jet_mask = jet_mask & jet_pu_mask()
        if min_drcut: jet_mask = jet_mask & (tree["jet_min_dr"] > min_drcut)
        if qglcut: jet_mask = jet_mask & (tree["jet_qgl"] > qglcut)
        
    njets = ak.sum(jet_mask,axis=-1)
    if njetcut: event_mask = event_mask & (njets >= njetcut)
    if njetcut_invert: event_mask = event_mask & (njets < njetcut_invert)
    if exclude_events_mask is not None: event_mask = event_mask & exclude_events_mask
    return event_mask,jet_mask

def xmass_selected_signal(tree,jets_index,njets=6,invm=700):
    top_jets_index, _ = get_top_njet_index(tree,jets_index,njets=njets)
    
    jet_pt = tree["jet_pt"]
    jet_m = tree["jet_m"]
    jet_eta = tree["jet_eta"]
    jet_phi = tree["jet_phi"]

    comb_jets_index = ak.combinations(top_jets_index,6)
    build_p4 = lambda index : vector.obj(pt=jet_pt[index],mass=jet_m[index],eta=jet_eta[index],phi=jet_phi[index])
    jet0, jet1, jet2, jet3, jet4, jet5 = [build_p4(jet) for jet in ak.unzip(comb_jets_index)]
    x_invm = (jet0+jet1+jet2+jet3+jet4+jet5).mass
    signal_comb = ak.argmin(np.abs(x_invm - invm),axis=-1)
    comb_mask = get_jet_index_mask(comb_jets_index,signal_comb[:,np.newaxis])
    jets_selected_index = ak.concatenate(ak.unzip(comb_jets_index[comb_mask]),axis=-1)
    
    selected_compare, _ = ak.broadcast_arrays(jets_selected_index[:,np.newaxis],jets_index)
    jets_remaining_index = jets_index[ ak.sum(jets_index==selected_compare,axis=-1)==0 ]
    
    return jets_selected_index,jets_remaining_index

def com_boost_vector(jet_pt,jet_eta,jet_phi,jet_m,njet=-1):
    """
    Calculate the COM boost vector for the top njets
    """
    if njet == -1: njet = ak.max(ak.count(jet_pt,axis=-1))
    fill_zero = lambda arr : ak.fill_none( ak.pad_none(arr,njet,axis=-1),0 )
    
    jet_pt = fill_zero(jet_pt)
    jet_eta = fill_zero(jet_eta)
    jet_phi = fill_zero(jet_phi)
    jet_m = fill_zero(jet_m)
        
    jet_vectors = [ vector.obj(pt=jet_pt[:,i],eta=jet_eta[:,i],phi=jet_phi[:,i],m=jet_m[:,i]) for i in range(njet) ]
    boost = jet_vectors[0]
    for jet_vector in jet_vectors[1:]: boost = boost + jet_vector
        
    return boost

def calc_y23(jet_pt):
    """
    measure of the third-jet pT relative to the summed transverse momenta
    of the two leading jets in a multi-jet event
    """
    ht2 = ak.sum(jet_pt[:,:2],axis=-1)
    pt3 = jet_pt[:,3]
    y23 = pt3**2/ht2**2
    
    return dict(event_y23=y23)

def calc_momentum_tensor(jet_px,jet_py,jet_pz):
    trace = ak.sum(jet_px**2+jet_py**2+jet_pz**2,axis=-1)
    Mij = lambda jet_pi,jet_pj : ak.sum(jet_pi*jet_pj,axis=-1)/trace

    a = Mij(jet_px,jet_px)
    b = Mij(jet_py,jet_py)
    c = Mij(jet_pz,jet_pz)
    d = Mij(jet_px,jet_py)
    e = Mij(jet_px,jet_pz)
    f = Mij(jet_py,jet_pz)

    m1 = ak.concatenate([a[:,np.newaxis],d[:,np.newaxis],e[:,np.newaxis]],axis=-1)
    m2 = ak.concatenate([d[:,np.newaxis],b[:,np.newaxis],f[:,np.newaxis]],axis=-1)
    m3 = ak.concatenate([e[:,np.newaxis],f[:,np.newaxis],c[:,np.newaxis]],axis=-1)
    M = ak.to_numpy(ak.concatenate([m1[:,np.newaxis],m2[:,np.newaxis],m3[:,np.newaxis]],axis=-2))
    return M

def calc_sphericity(jet_pt,jet_eta,jet_phi,jet_m,njet=-1):
    """
    Calculate sphericity/aplanarity in the COM frame of the top njets
    
    Sphericity: Measures how spherical the event is
    0 -> Spherical | 1 -> Collimated
    
    Aplanarity: Measures the amount of transverse momentum in or out of the jet plane
    """
    
    boost = com_boost_vector(jet_pt,jet_eta,jet_phi,jet_m,njet)
    boosted_jets = vector.obj(pt=jet_pt,eta=jet_eta,phi=jet_phi,m=jet_m).boost_p4(-boost)
    jet_px,jet_py,jet_pz = boosted_jets.px,boosted_jets.py,boosted_jets.pz
    
    M = calc_momentum_tensor(jet_px,jet_py,jet_pz)
    
    eig_w,eig_v = np.linalg.eig(M)
    eig_w = np.abs(eig_w) # make sure we only look at the absolue magnitude of values
    eig_w = eig_w/np.sum(eig_w,axis=-1)[:,np.newaxis] # make sure eigenvalues are normalized
    eig_w = ak.sort(eig_w)

    eig_w1 = eig_w[:,2]
    eig_w2 = eig_w[:,1]
    eig_w3 = eig_w[:,0]

    S = 3/2 * (eig_w2+eig_w3)
    St= 2* eig_w2 / ( eig_w1 + eig_w2)
    A = 3/2 * eig_w3
    F = eig_w2/eig_w1
    
    return dict(M_eig_w1=eig_w1,M_eig_w2=eig_w2,M_eig_w3=eig_w3,event_S=S,event_St=St,event_A=A,event_F=F)

def find_thrust_phi(jet_px,jet_py,tol=1e-05,niter=10,gr=(1+np.sqrt(5))/2):
    """
    Maximizing thrust via golden-selection search
    """
    
    jet_ones = ak.ones_like(jet_px)
    
    a = -(np.pi/2)*jet_ones
    b =  (np.pi/2)*jet_ones
    
    c = b-(b-a)/gr
    d = a+(b-a)/gr
    
    f = lambda phi : -ak.sum(np.abs(jet_px*np.cos(phi)+jet_py*np.sin(phi)),axis=-1)

    it = 0
    while ak.all( np.abs(b - a) > tol ) and it < niter:
        is_c_low = f(c) < f(d)
        is_d_low = is_c_low == False
        
        b = b*(is_d_low) + d*(is_c_low)
        a = c*(is_d_low) + a*(is_c_low)
        
        c = b-(b-a)/gr
        d = a+(b-a)/gr

        it += 1
        
    return (b[:,0]+a[:,0])/2

def calc_thrust(jet_pt,jet_eta,jet_phi,jet_m):
    """
    The total thrust of the jets in the event
    """
    
    boost = com_boost_vector(jet_pt,jet_eta,jet_phi,jet_m)
    boosted_jets = vector.obj(pt=jet_pt,eta=jet_eta,phi=jet_phi,m=jet_m).boost_p4(-boost)
    jet_pt,jet_eta,jet_phi,jet_m = boosted_jets.pt,boosted_jets.eta,boosted_jets.phi,boosted_jets.m
    
    jet_ht = ak.sum(jet_pt,axis=-1)
    jet_vectors = vector.obj(pt=jet_pt,eta=jet_eta,phi=jet_phi,m=jet_m)
    jet_px,jet_py = jet_vectors.px,jet_vectors.py
    
    thrust_phi = find_thrust_phi(jet_px,jet_py)
    Tt = 1 - ak.sum(np.abs(jet_px*np.cos(thrust_phi)+jet_py*np.sin(thrust_phi)),axis=-1)/jet_ht
    Tm = ak.sum(np.abs(jet_px*np.sin(thrust_phi)-jet_py*np.cos(thrust_phi)),axis=-1)/jet_ht
    
    return dict(thrust_phi=thrust_phi,event_Tt=Tt,event_Tm=Tm)

def calc_asymmetry(jet_pt,jet_eta,jet_phi,jet_m,njet=-1):
    """
    Calculate the asymmetry of the top njets in their COM frame
    """
    
    boost = com_boost_vector(jet_pt,jet_eta,jet_phi,jet_m,njet)
    boosted_jets = vector.obj(pt=jet_pt,eta=jet_eta,phi=jet_phi,m=jet_m) #.boost_p4(-boost)
    jet_px,jet_py,jet_pz = boosted_jets.px,boosted_jets.py,boosted_jets.pz
    
    jet_p = np.sqrt(jet_px**2+jet_py**2+jet_pz**2)
    
    AL = ak.sum(jet_pz,axis=-1)/ak.sum(jet_p,axis=-1)
    
    return dict(event_AL=AL)

def optimize_var_cut(selections,variable,varmin=None,varmax=None,method=min,plot=False):
    varmin = min([ ak.min(selection[variable]) for selection in selections ]) if varmin == None else varmin
    varmax = max([ ak.max(selection[variable]) for selection in selections ]) if varmax == None else varmax
    
    if method is min: method = lambda arr,cut : arr < cut
    elif method is max: method = lambda arr,cut : arr > cut

    def function(cut):
        nevents = [ ak.sum( selection["scale"][method(selection[variable],cut)] ) for selection in selections ]
        bkg_eff = sum(nevents[1:])/sum([ ak.sum(selection["scale"]) for selection in selections[1:]])
        bovers = sum(nevents[1:])/nevents[0] if nevents[0] != 0 else 0
        score = bovers*bkg_eff
        return -score

    if plot:

        x = np.linspace(varmin,varmax,100)
        y = np.vectorize(function)(x)

        graph_simple(x,-y,xlabel=f'{variable} cut',ylabel="b/s*b_eff",marker=None)
    
    f_min = scipy.optimize.fmin(function,(varmax+varmin)/2)
    return f_min
