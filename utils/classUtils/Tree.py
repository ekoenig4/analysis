from . import *

import glob

def init_file(self,tfname):
    self.tfname = tfname
    self.tfile = ut.open(tfname)
    self.total_events = ak.sum(self.tfile["n_events"].to_numpy(),axis=None)
    self.ttree = self.tfile["sixBtree"].arrays()
    return ak.count(self.ttree) > 0

def init_dir(self,tdir):
    self.tfname = f"{tdir}/ntuple_*.root"
    filelist = glob.glob(self.tfname)

    self.tfiles = [ ut.open(fname) for fname in filelist ]
    self.total_events = sum([ak.sum(tfile["n_events"].to_numpy(),axis=None) for tfile in self.tfiles])
    ttrees = list(ut.iterate(self.tfname,allow_missing=True))
    if len(ttrees) == 0: return False
    
    self.ttree = ak.concatenate(ttrees)
    return ak.count(self.ttree) > 0

class Tree:
    def __init__(self,tfname):
        if os.path.isdir(tfname): valid = init_dir(self,tfname)
        else:                    valid = init_file(self,tfname)
        self.valid = valid
        if not valid: return
        
        self.extended = {}
        self.nevents = ak.size( self["Run"] )
        self.is_signal = "NMSSM" in self.tfname

        self.sample,self.xsec = next( ((key,value) for key,value in xsecMap.items() if key in tfname),("unk",1) )
        self.scale = self.xsec / self.total_events
        
        self.all_events_mask = ak.ones_like(self["Run"]) == 1
        self.all_jets_mask = ak.ones_like(self["jet_pt"]) == 1
        
        self.sixb_jet_mask = self["jet_signalId"] != -1
        self.bkgs_jet_mask = self.sixb_jet_mask == False

        self.sixb_found_mask = self["nfound_all"] == 6
        # self.reco_XY()
    def __str__(self):
        string = [
            f"=== File Info ===",
            f"File: {self.tfname}",
            f"Total Events:    {self.total_events}",
            f"Selected Events: {self.nevents}",
        ]
        return "\n".join(string)
    def __getitem__(self,key): 
        if key in self.extended:
            return self.extended[key]
#         if key == "jet_pt": key = "jet_ptRegressed"
        return self.ttree[key]
    def scale_weights(self,jets=False):
        if jets:
            njets = ak.sum(self.all_jets_mask,axis=None)
            weights = np.full(shape=njets,fill_value=self.scale,dtype=np.float)
            return ak.unflatten(weights,ak.sum(self.all_jets_mask,axis=-1))
        
        return np.full(shape=self.nevents,fill_value=self.scale,dtype=np.float)
    def reco_XY(self):
        bjet_p4 = lambda key : vector.obj(pt=self[f"gen_{key}_recojet_pt"],eta=self[f"gen_{key}_recojet_eta"],
                                          phi=self[f"gen_{key}_recojet_phi"],mass=self[f"gen_{key}_recojet_m"])
        hx_b1 = bjet_p4("HX_b1")
        hx_b2 = bjet_p4("HX_b2")
        hy1_b1= bjet_p4("HY1_b1")
        hy1_b2= bjet_p4("HY1_b2")
        hy2_b1= bjet_p4("HY2_b1")
        hy2_b2= bjet_p4("HY2_b2")
        
        Y = hy1_b1 + hy1_b2 + hy2_b1 + hy2_b2
        X = hx_b1 + hx_b2 + Y
        
        self.extended.update({"X_pt":X.pt,"X_m":X.mass,"X_eta":X.eta,"X_phi":X.phi,
                              "Y_pt":Y.pt,"Y_m":Y.mass,"Y_eta":Y.eta,"Y_phi":Y.phi})
    def calc_jet_dr(self,compare=None,tag="jet"):
        select_eta = self["jet_eta"]
        select_phi = self["jet_phi"]

        compare_eta = self["jet_eta"][compare]
        compare_phi = self["jet_phi"][compare]
        
        dr = calc_dr(select_eta,select_phi,compare_eta,compare_phi)
        dr_index = ak.local_index(dr,axis=-1)

        remove_self = dr != 0
        dr = dr[remove_self]
        dr_index = dr_index[remove_self]

        imin_dr = ak.argmin(dr,axis=-1,keepdims=True)
        imax_dr = ak.argmax(dr,axis=-1,keepdims=True)

        min_dr = ak.flatten(dr[imin_dr],axis=-1)
        imin_dr = ak.flatten(dr_index[imin_dr],axis=-1)

        max_dr = ak.flatten(dr[imax_dr],axis=-1)
        imax_dr = ak.flatten(dr_index[imax_dr],axis=-1)

        self.extended.update({f"{tag}_min_dr":min_dr,f"{tag}_imin_dr":imin_dr,f"{tag}_max_dr":max_dr,f"{tag}_imax_dr":imax_dr})

class TreeList:
    def __init__(self,filelist):
        self.filelist = filelist
        self.collection = []
        for fname in filelist:
            tree = Tree(fname)
            if tree.valid:
                self.collection.append(tree)
            
    def __iter__(self): return iter(self.collection)
                
    def __getitem__(self,key):
        print("getitem",key)
        if type(key) == int: return self.collection[key]
        return ak.concatenate([ tree[key] for tree in self ])

    def scale_weights(self,jets=False):
        return ak.concatenate([ tree.scale_weights(jets=jets) for tree in self ])
        
