from . import *

import os

class Selection(Tree):
    def __init__(self,tree,cuts={},include=None,previous=None,variable=None,njets=-1,mask=None,tag=None,ignore_tag=False):
        copy_fields(tree,self)
        self.extended = {}
        
        self.tree = tree
        self.subset = "selected"
        
        self.include = include
        self.previous = previous
        self.ignore_previous_tag = ignore_tag
        self.ignore_include_tag = ignore_tag
            
        self.previous_index = previous.total_jets_selected_index if previous else None
        self.previous_selected = previous.total_jets_selected if previous else (self.all_jets_mask == False)
        self.previous_njets = previous.total_njets if previous else 0
        
        self.include_jet_mask = include.total_jets_selected if include else self.all_jets_mask
        self.include_events_mask = include.mask if include else self.all_events_mask
        
        self.exclude_jet_mask = previous.total_jets_selected if previous else None
        self.exclude_events_mask = previous.mask if previous else self.all_events_mask
        
        self.previous_events_mask = self.include_events_mask & self.exclude_events_mask
        self.previous_nevents = ak.sum(self.previous_events_mask)
        
        if cuts is None: cuts = {"passthrough":True}
        self.cuts = cuts
        self.mask = self.all_events_mask
        self.jets_passed = self.all_jets_mask
        
        self.choose_jets(cuts,variable,njets,mask,tag)

    def __getitem__(self,key):
        item = self.get(key)
        return item[self.mask]

    def get(self,key):
        item = self.extended[key] if key in self.extended else self.tree[key]
        if key.startswith("jet_"): item = item[self.jets_selected_index]
        return item

    def scale_weights(self,jets=False):
        weights = self.tree.scale_weights(jets=jets)
        if jets: weights = weights[self.jets_selected]
        return weights[self.mask]
                
    def choose_jets(self,cuts={},variable=None,njets=-1,mask=None,tag=None):
        if cuts is None: cuts = {"passthrough":True}
        if any(cuts): 
            self.cuts = dict(**self.cuts)
            self.cuts["passthrough"] = False
            self.cuts.update(cuts)
        if tag: self.tag = tag
        self.mask, self.jets_passed = std_preselection(self.tree,exclude_events_mask=self.previous_events_mask & self.mask,
                                                        exclude_jet_mask=self.exclude_jet_mask,
                                                        include_jet_mask=self.include_jet_mask & self.jets_passed,**self.cuts)
        if mask is not None: self.mask = self.mask & mask
        
        self.njets_passed = ak.sum(self.jets_passed,axis=-1)
        self.jets_failed = exclude_jets(self.tree.all_jets_mask,self.jets_passed)
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
            self.jets_passed_index = sort_jet_index(self.tree,self.variable,self.jets_passed,method=method)
        self.select_njets(njets)
        
    def sorted_jets(self,variable,njets=-1,method=max):
        new_selection = self.copy()
        new_selection.sort_jets(variable,njets,method)
        return new_selection
        
    def select_njets(self,njets):
        self.extra_collections = False
        
        self.njets = njets
        self.jets_selected_index, self.jets_remaining_index = get_top_njet_index(self.tree,self.jets_passed_index,self.njets)
        
        self.jets_selected = get_jet_index_mask(self.tree,self.jets_selected_index)
        self.njets_selected = ak.sum(self.jets_selected,axis=-1)
        self.extended["n_jet"] = self.njets_selected
        
        self.total_jets_selected_index = self.jets_selected_index
        if self.previous: self.total_jets_selected_index = ak.concatenate([self.previous_index,self.total_jets_selected_index],axis=-1)
        self.total_jets_selected = self.previous_selected | self.jets_selected
        self.total_njets = self.previous_njets + (self.njets if self.njets != -1 else 6)

        if self.is_signal: self.build_extra_collections()
        
    def selected_njets(self,njets):
        new_selection = self.copy()
        new_selection.select_njets(njets)
        return new_selection
    
    def sort_selected_jets(self,variable):
        self.variable = variable
        self.jets_selected_index = sort_jet_index(self.tree,self.variable,self.jets_selected)
        
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
        jet_pt = fill_zero(self["jet_pt"])
        jet_eta = fill_zero(self["jet_eta"])
        jet_phi = fill_zero(self["jet_phi"])
        jet_m = fill_zero(self["jet_m"])
        
        ijet_p4 = [ vector.obj(pt=jet_pt[:,ijet],eta=jet_eta[:,ijet],phi=jet_phi[:,ijet],mass=jet_m[:,ijet]) for ijet in range(6) ]
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
        
        return title
    
    def __str__(self):
        return f"--- {self.title()} ---\n{self.score()}"
           
class SelectionScore:
    def __init__(self,selection):
        tree = selection.tree
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
            nsixb_selected = selection.nsixb_selected[mask]
            nsixb_passed = selection.nsixb_passed[mask]
            nbkgs_passed = selection.nbkgs_passed[mask]
            
            nsixb_total = ak.sum(tree.sixb_jet_mask[mask],axis=-1)
            nbkgs_total = ak.sum(tree.bkgs_jet_mask[mask],axis=-1)
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
        copy_fields(selection,self)
        self.extended = dict(**self.extended)
        
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
        self.jets_failed = exclude_jets(self.tree.all_jets_mask,self.jets_passed)
        self.njets_failed = ak.sum(self.jets_failed,axis=-1)
        
        self.extended["n_jet"] = self.njets_selected
        
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
