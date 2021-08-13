from . import *

import glob

def add_sample(self,fname,cutflow,cutflow_labels,ttree,sample,xsec,scale):
    self.nfiles += 1
    self.filenames.append(fname)
    self.cutflow.append(cutflow)
    self.total_events.append(cutflow[0])
    ncutflow = len(cutflow_labels)
    if ncutflow > self.ncutflow:
        self.ncutflow = ncutflow
        self.cutflow_labels = cutflow_labels
    self.raw_events.append(len(ttree))
    self.ttrees.append(ttree)
    self.samples.append(sample)
    self.xsecs.append(xsec)
    self.scales.append(scale)

def init_file(self,tfname):
    cutflow = ut.open(tfname+":h_cutflow")
    cutflow_labels = cutflow.axis().labels()
    if cutflow_labels is None: cutflow_labels = []
    
    cutflow = cutflow.to_numpy()[0]
    
    if not self.lazy:
        ttree = ut.open(tfname+":sixBtree").arrays()
    else:
        ttree = ut.lazy(tfname+":sixBtree")

    valid = len(ttree) > 0

    if not valid and self.verify: return
    sample,xsec = next( ((key,value) for key,value in xsecMap.items() if key in tfname),("unk",1) )
    scale = xsec / cutflow[0] if type(xsec) == float else 1

    add_sample(self,tfname,cutflow,cutflow_labels,ttree,sample,xsec,scale)
    
class Tree:
    def __init__(self,filenames,verify=True,lazy=False):
        if type(filenames) != list: filenames = [filenames]
        self.verify = verify
        self.lazy = lazy
        
        self.nfiles = 0
        self.filenames = []
        self.tfiles = []
        self.cutflow = []
        self.cutflow_labels = []
        self.ncutflow = 0
        self.total_events = []
        self.raw_events = []
        self.ttrees = []
        self.samples = []
        self.xsecs = []
        self.scales = []
        
        for fname in filenames: self.addTree(fname)
        self.valid = self.nfiles > 0

        if not self.valid: return
        self.is_data = any("Data" in fn for fn in self.filenames)

        sample_tag = [ next( (tag for key,tag in tagMap.items() if key in sample),None ) for sample in self.samples ]

        if (sample_tag.count(sample_tag[0]) == len(sample_tag)): self.tag = sample_tag[0]
        else: self.tag = "Bkg"
        
        if self.is_data: self.tag = "Data"
        self.color = colorMap.get(self.tag,None)

        self.cutflow = [ ak.fill_none( ak.pad_none(cutflow,self.ncutflow,axis=0,clip=True),0 ) for cutflow in self.cutflow ]
        
        if not self.lazy:
            self.ttree = ak.concatenate(self.ttrees)

        sample_id = ak.concatenate([ ak.Array([i]*len(tree)) for i,tree in enumerate(self.ttrees) ])
        
        self.extended = {"sample_id":sample_id}
        self.nevents = sum(self.raw_events)
        self.is_signal = all("NMSSM" in fname for fname in self.filenames)
        self.build_scale_weights()
        
        self.all_events_mask = ak.Array([True]*self.nevents)
        njet = self["n_jet"]
        self.all_jets_mask = ak.unflatten( np.repeat(np.array(self.all_events_mask,dtype=bool),njet),njet )

        self.mask = self.all_events_mask
        self.jets_selected = self.all_jets_mask
        
        if self.lazy: return
        
        self.sixb_jet_mask = self["jet_signalId"] != -1
        self.bkgs_jet_mask = self.sixb_jet_mask == False

        self.sixb_found_mask = self["nfound_presel"] == 6

        # self.reco_XY()
    def __str__(self):
        if not self.valid: return "invalid"
        sample_string = [
            f"=== File Info ===",
            f"File: {self.filenames}",
            f"Total Events:    {self.total_events}",
            f"Raw Events:      {self.raw_events}",
            f"Selected Events: {self.nevents}",
        ]
        return "\n".join(sample_string)
    def __getitem__(self,key): 
        if key in self.extended:
            return self.extended[key]
        item = self.ttree[key] if not self.lazy else ak.concatenate([tree[key] for tree in self.ttrees])
        self.extend(key=item)
        return item
    def get(self,key): return self[key]
    def addTree(self,fname): init_file(self,fname)
    def extend(self,**kwargs): self.extended.update(**kwargs)
    def build_scale_weights(self):
        event_scale = ak.concatenate( [np.full(shape=len(tree),fill_value=scale,dtype=np.float) for scale,tree in zip(self.scales,self.ttrees)] )
        jet_scale = ak.unflatten( np.repeat(ak.to_numpy(event_scale),self["n_jet"]),self["n_jet"] )
        self.extend(scale=event_scale,jet_scale=jet_scale)
        
        if "n_higgs" in ak.fields(self.ttrees[0]):
            higgs_scale = ak.unflatten( np.repeat(ak.to_numpy(event_scale),self["n_higgs"]),self["n_higgs"] )
            self.extend(higgs_scale=higgs_scale)
    
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
        select_eta = self.get("jet_eta")
        select_phi = self["jet_phi"]

        if compare is None: compare = self.jets_selected
        
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

    def calc_event_shapes(self):
        jet_pt,jet_eta,jet_phi,jet_m = self.get("jet_pt"),self.get("jet_eta"),self.get("jet_phi"),self.get("jet_m")
        
        self.extend (
            **calc_y23(jet_pt),
            **calc_sphericity(jet_pt,jet_eta,jet_phi,jet_m),
            **calc_thrust(jet_pt,jet_eta,jet_phi,jet_m),
            **calc_asymmetry(jet_pt,jet_eta,jet_phi,jet_m),
        )

    def calc_btagsum(self):
        for nj in (5,6): self.extend(**{f"jet{nj}_btagsum":ak.sum(self.get("jet_btag")[:,:nj],axis=-1)})
        
    def copy(self):
        new_tree = CopyTree(self)
        return new_tree
        
class CopyTree(Tree):
    def __init__(self,tree):
        copy_fields(tree,self)
