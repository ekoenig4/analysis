from . import *

class Branches:
    def __init__(self,tfname,sample=None,xsec=1,is_signal=False):
        self.tfname = tfname
        self.tfile = ut.open(tfname)
        self.total_events = ak.sum(self.tfile["n_events"].to_numpy(),axis=None)
        self.ttree = self.tfile["sixBtree"]
        self.extended = {}
        self.nevents = ak.size( self["Run"] )
        self.is_signal = is_signal

        self.sample = sample
        self.xsec = xsec
        if sample: self.xsec = xsecMap[sample]
        self.scale = self.xsec/self.total_events
        
        self.all_events_mask = ak.broadcast_arrays(True,self["Run"])[0]
        self.all_jets_mask = ak.broadcast_arrays(True,self["jet_pt"])[0]
        
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
        return self.ttree[key].array()
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
class Selection:
    def __init__(self,branches,cuts={},include=None,previous=None,variable=None,njets=-1,mask=None,tag="selected",ignore_tag=False):
        self.tag = tag
        self.branches = branches
        self.is_signal = branches.is_signal
        self.scale = branches.scale
        
        self.include = include
        self.previous = previous
        self.ignore_previous_tag = ignore_tag
        self.ignore_include_tag = ignore_tag
            
        self.previous_index = previous.total_jets_selected_index if previous else None
        self.previous_selected = previous.total_jets_selected if previous else (branches.all_jets_mask == False)
        self.previous_njets = previous.total_njets if previous else 0
        
        self.include_jet_mask = include.total_jets_selected if include else branches.all_jets_mask
        self.include_events_mask = include.mask if include else branches.all_events_mask
        
        self.exclude_jet_mask = previous.total_jets_selected if previous else None
        self.exclude_events_mask = previous.mask if previous else branches.all_events_mask
        
        self.previous_events_mask = self.include_events_mask & self.exclude_events_mask
        self.previous_nevents = ak.sum(self.previous_events_mask)
        
        self.sixb_jet_mask = branches.sixb_jet_mask
        self.bkgs_jet_mask = branches.bkgs_jet_mask
        
        if cuts is None: cuts = {"passthrough":True}
        self.cuts = cuts
        self.mask = branches.all_events_mask
        self.jets_passed = branches.all_jets_mask
        
        self.choose_jets(cuts,variable,njets,mask)
                
    def choose_jets(self,cuts={},variable=None,njets=-1,mask=None,tag=None):
        if cuts is None: cuts = {"passthrough":True}
        if any(cuts): 
            self.cuts = dict(**self.cuts)
            self.cuts["passthrough"] = False
            self.cuts.update(cuts)
        if tag: self.tag = tag
            
        self.mask, self.jets_passed = std_preselection(self.branches,exclude_events_mask=self.previous_events_mask & self.mask,
                                                        exclude_jet_mask=self.exclude_jet_mask,
                                                        include_jet_mask=self.include_jet_mask & self.jets_passed,**self.cuts)
        if mask is not None: self.mask = self.mask & mask
        self.njets_passed = ak.sum(self.jets_passed,axis=-1)
        self.jets_failed = exclude_jets(self.branches.all_jets_mask,self.jets_passed)
        self.njets_failed = ak.sum(self.jets_failed,axis=-1)
        
        self.nevents = ak.sum(self.mask)
        self.sort_jets(variable,njets)
                
    def chosen_jets(self,cuts={},variable=None,njets=-1,mask=None,tag=None):
        new_selection = self.copy()
        new_selection.choose_jets(cuts,variable,njets,mask,tag)
        return new_selection
        
    def sort_jets(self,variable,njets=-1,method=max):
        self.variable = variable
        
        if variable is None and self.include:
            included_passed_index = self.jets_passed[self.include.total_jets_selected_index]
            self.jets_passed_index = self.include.total_jets_selected_index[included_passed_index]
        else: 
            self.jets_passed_index = sort_jet_index(self.branches,self.variable,self.jets_passed,method=method)
        self.select_njets(njets)
        
    def sorted_jets(self,variable,njets=-1,method=max):
        new_selection = self.copy()
        new_selection.sort_jets(variable,njets,method)
        return new_selection
        
    def select_njets(self,njets):
        self.extra_collections = False
        
        self.njets = njets
        self.jets_selected_index, self.jets_remaining_index = get_top_njet_index(self.branches,self.jets_passed_index,self.njets)
        
        self.jets_selected = get_jet_index_mask(self.branches,self.jets_selected_index)
        self.njets_selected = ak.sum(self.jets_selected,axis=-1)
        
        self.total_jets_selected_index = self.jets_selected_index
        if self.previous: self.total_jets_selected_index = ak.concatenate([self.previous_index,self.total_jets_selected_index],axis=-1)
        self.total_jets_selected = self.previous_selected | self.jets_selected
        self.total_njets = self.previous_njets + (self.njets if self.njets != -1 else 6)
        
    def selected_njets(self,njets):
        new_selection = self.copy()
        new_selection.select_njets(njets)
        return new_selection
    
    def sort_selected_jets(self,variable):
        self.variable = variable
        self.jets_selected_index = sort_jet_index(self.branches,self.variable,self.jets_selected)
        
    def sorted_selected_jets(self,variable):
        new_selection = self.copy()
        new_selection.sort_selected_jets(variable)
        return new_selection
    
    def masked(self,mask):
        new_selection = self.copy()
        new_selection.mask = new_selection.mask & mask
        new_selection.nevents = ak.sum(new_selection.mask)
        return new_selection
    
    def reco_X(self):
        fill_zero = lambda ary : ak.fill_none(ak.pad_none(ary,6,axis=-1,clip=1),0)
        
        jets = self.jets_selected_index
        jet_pt = fill_zero(self.branches["jet_pt"][jets])
        jet_eta = fill_zero(self.branches["jet_eta"][jets])
        jet_phi = fill_zero(self.branches["jet_phi"][jets])
        jet_m = fill_zero(self.branches["jet_m"][jets])

        njets = min(self.njets,6) if self.njets != -1 else 6
        
        ijet_p4 = [ vector.obj(pt=jet_pt[:,ijet],eta=jet_eta[:,ijet],phi=jet_phi[:,ijet],mass=jet_m[:,ijet]) for ijet in range(njets) ]
        X_reco = ijet_p4[0]+ijet_p4[1]+ijet_p4[2]+ijet_p4[3]+ijet_p4[4]+ijet_p4[5]
        return {"m":X_reco.mass,"pt":X_reco.pt,"eta":X_reco.eta,"phi":X_reco.phi}

    def build_extra_collections(self):
        if self.extra_collections: return
        self.extra_collections = True

        def build_extra_jet(tag,jet_mask):
            setattr(self,f"{tag}_passed",self.jets_passed & jet_mask)
            setattr(self,f"n{tag}_passed",ak.sum(getattr(self,f"{tag}_passed"),axis=-1))
            
            setattr(self,f"{tag}_passed_position", get_jet_position(self.jets_passed_index,jet_mask))
            setattr(self,f"{tag}_passed_index", self.jets_passed_index[getattr(self,f"{tag}_passed_position")])
            
            setattr(self,f"{tag}_failed",self.jets_failed & jet_mask)
            setattr(self,f"n{tag}_failed",ak.sum(getattr(self,f"{tag}_failed"),axis=-1))
            
            setattr(self,f"{tag}_selected", self.jets_selected & jet_mask)
            setattr(self,f"n{tag}_selected",ak.sum(getattr(self,f"{tag}_selected"),axis=-1))
            
            setattr(self,f"{tag}_selected_position", get_jet_position(self.jets_selected_index,jet_mask))
            setattr(self,f"{tag}_selected_index", self.jets_selected_index[getattr(self,f"{tag}_selected_position")])
        build_extra_jet("sixb",self.sixb_jet_mask)
        build_extra_jet("bkgs",self.bkgs_jet_mask)
        
    def score(self): 
        return SelectionScore(self)
    
    def merge(self,tag=None):
        return MergedSelection(self,tag=tag)
    
    def copy(self):
        return CopySelection(self)
    
    def title(self,i=0):
        ignore = lambda tag : any( _ in tag for _ in ["baseline","preselection"] )
        if self.tag is None: return
        title = f"{self.njets} {self.tag}" if self.njets != -1 else f"all {self.tag}"
        variable = self.variable if self.variable else "jet_pt"
        if variable != "jet_pt": title = f"{title} / {variable.replace('jet_','')}"
        if self.include and self.include.tag and not ignore(self.include.tag) and not self.ignore_include_tag: 
            title = f"{self.include.title(1)} & {title}"
            if i != 1: title = f"({title})" 
        if self.previous and self.previous.tag and not ignore(self.previous.tag) and not self.ignore_previous_tag: 
            title = f"{self.previous.title(2)} | {title}"
            
        if i != 0: return title
        
#         tag_map = {}
#         for tag in re.split('\s[&|]\s',title): 
#             tag = re.sub('[\(\)]','',tag)
#             if tag not in tag_map: tag_map[tag] = f"a{len(tag_map)}"
#         tag_eq = title.replace(" & ","*").replace(" | ","+")#.replace(" / ","/")
#         for tag,var in tag_map.items(): tag_eq = tag_eq.replace(tag,var)
#         tag_eq_sim = str(sp.simplify(tag_eq)).replace(" ","")
#         tag_sim = tag_eq_sim.replace("*"," & ").replace("+"," | ")#.replace("/"," / ")
#         for tag,var in tag_map.items(): tag_sim = tag_sim.replace(var,tag)
        
        return title
    
    def __str__(self):
        return f"--- {self.title()} ---\n{self.score()}"
           
class SelectionScore:
    def __init__(self,selection):
        branches = selection.branches
        mask = selection.mask
        
        njets = selection.njets
        if njets < 0: njets = 6
        self.nsixb = min(6,njets)
        
        nevents = ak.sum(selection.mask)
        njets_selected = selection.njets_selected[mask]
        njets_passed = selection.njets_passed[mask]
        
        self.efficiency = nevents/selection.previous_nevents
        self.prompt_list = ["Event Efficiency:   {efficiency:0.2}",]

        if selection.is_signal:
            selection.build_extra_collections()
            self.purity = ak.sum(nsixb_selected == self.nsixb)/nevents
            self.jet_sovert = ak.sum(nsixb_passed)/ak.sum(njets_passed)
            self.jet_soverb = ak.sum(nsixb_passed)/ak.sum(nbkgs_passed)
            self.jet_misstr = ak.sum(nbkgs_passed)/ak.sum(nbkgs_total)
            self.jet_eff    = ak.sum(nsixb_passed)/ak.sum(nsixb_total)
        
            self.prompt_list = [
                "Event Efficiency:   {efficiency:0.2}",
                "Selected Purity({nsixb}): {purity:0.2f}",
                "Passed Jet S/T:     {jet_sovert:0.2f}",
                #             "Passed Jet MR:      {jet_misstr:0.2f}",
                #             "Passed Jet Eff:     {jet_eff:0.2f}",
            ]
    def __str__(self):
        prompt = '\n'.join(self.prompt_list)
        return prompt.format(**vars(self))
    def savetex(self,fname):
        output = '\\\\ \n'.join(self.prompt_list).format(**vars(self))
        with open(f"{fname}.tex","w") as f: f.write(output)
            
class CopySelection(Selection):
    def __init__(self,selection):
        for key,value in vars(selection).items():
            setattr(self,key,value)
        
class MergedSelection(CopySelection): 
    def __init__(self,selection,tag="merged selection"): 
        CopySelection.__init__(self,selection)

        previous = selection.previous
        while(previous is not None): 
            self.add(previous)
            self.previous_nevents = previous.previous_nevents
            previous = previous.previous
        
        self.jets_passed_index = ak.concatenate((self.jets_selected_index,self.jets_remaining_index),axis=-1)
        self.njets_passed = ak.sum(self.jets_passed,axis=-1)
        self.njets_selected = ak.sum(self.jets_selected,axis=-1)
        self.jets_failed = exclude_jets(self.branches.all_jets_mask,self.jets_passed)
        self.njets_failed = ak.sum(self.jets_failed,axis=-1)
        
        self.previous = None
        self.last = selection
        if self.njets != -1: self.njets = self.total_njets
        
        self.ignore_previous_tag = True
        self.tag = tag
        self.variable = None
        
    def add(self,selection):
        for key in ("jets_passed","jets_selected"):
            setattr(self,key, getattr(self,key) | getattr(selection,key))
        for key in ("jets_selected_index",):
            setattr(self,key, ak.concatenate((getattr(selection,key),getattr(self,key)),axis=-1))
        for key in ("jets_failed",):
            setattr(self,key, getattr(self,key) & getattr(selection,key))
    
class SignalSelection(MergedSelection):
    signal_methods = {
        "xmass":xmass_selected_signal,
        "top":get_top_njet_index
    }
    def __init__(self,branches,previous,njets=6,mask=None,method="top"):
        MergedSelection.__init__(self,previous)
        
        self.variable = None
        
        self.previous = previous
        
        self.previous_index = self.jets_selected_index
        self.previous_njets = self.total_njets
        
        self.jets_order = self.jets_selected_index
        self.jets_passed = self.jets_selected
        self.njets_passed = ak.sum(self.jets_passed,axis=-1)
        
        self.previous_events_mask = self.previous.mask
        self.previous_nevents = ak.sum(self.previous_events_mask)
        
        if mask is not None: self.mask = self.mask & mask
        
        self.select_signal(njets,method)
        
    def select_signal(self,njets=6,method="top"):
        self.tag = f"{method} selected"
        self.njets = min(6,njets) if njets != -1 else 6
        if method == "top": njets = self.njets
        
        self.jets_selected_index, self.jets_remaining_index = self.signal_methods[method](self.branches,self.previous_index,njets=njets)
    
        self.jets_selected = get_jet_index_mask(self.branches,self.jets_selected_index)
        self.njets_selected = ak.sum(self.jets_selected,axis=-1)
        
        self.jets_remaining = get_jet_index_mask(self.branches,self.jets_remaining_index)
        self.njets_remaining = ak.sum(self.jets_remaining,axis=-1)
        
    def title(self,i=0): return Selection.title(self)
