from . import *

class Branches:
    def __init__(self,ttree):
        self.ttree = ttree
        self.extended = {}
        self.nevents = ak.size( self["Run"] )
        
        self.all_events_mask = ak.broadcast_arrays(True,self["Run"])[0]
        self.all_jets_mask = ak.broadcast_arrays(True,self["jet_pt"])[0]

        self.sixb_found_mask = self["nfound_all"] == 6
        self.nsignal = ak.sum( self.sixb_found_mask )
        self.sixb_jet_mask = get_jet_index_mask(self,self["signal_bjet_index"])
        self.reco_XY()
    def __getitem__(self,key): 
        if key in self.extended:
            return self.extended[key]
        return self.ttree[key].array()
    def reco_XY(self):
        bjet_p4 = lambda key : vector.obj(pt=self[f"gen_{key}_recojet_ptRegressed"],eta=self[f"gen_{key}_recojet_eta"],
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
    
class Selection:
    def __init__(self,branches,cuts={},include=None,previous=None,variable=None,njets=-1,mask=None,tag=""):
        self.tag = tag
        self.branches = branches
        
        self.include = include
        self.previous = previous
            
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
        self.cuts = cuts
        self.mask = branches.all_events_mask
        self.jets_captured = branches.all_jets_mask
        
        self.capture_jets(cuts,variable,njets,mask)
                
    def capture_jets(self,cuts={},variable=None,njets=-1,mask=None,tag=None):
        if any(cuts): 
            self.cuts = dict(**self.cuts)
            self.cuts["passthrough"] = False
            self.cuts.update(cuts)
        if tag: self.tag = tag
            
        self.mask, self.jets_captured = std_preselection(self.branches,exclude_events_mask=self.previous_events_mask & self.mask,
                                                         exclude_jet_mask=self.exclude_jet_mask,
                                                         include_jet_mask=self.include_jet_mask & self.jets_captured,**self.cuts)
        if mask is not None: self.mask = self.mask & mask
        self.njets_captured = ak.sum(self.jets_captured,axis=-1)
        
        self.sixb_captured = self.jets_captured & self.sixb_jet_mask
        self.nsixb_captured = ak.sum(self.sixb_captured,axis=-1)
        
        self.nevents = ak.sum(self.mask)
        self.nsignal = ak.sum(self.mask & self.branches.sixb_found_mask)
        
        self.sort_jets(variable,njets)
                
    def captured_jets(self,cuts={},variable=None,njets=-1,mask=None,tag=None):
        new_selection = self.copy()
        new_selection.capture_jets(cuts,variable,njets,mask,tag)
        return new_selection
        
    def sort_jets(self,variable,njets=-1,method=max):
        self.variable = variable
        self.jets_captured_index = sort_jet_index(self.branches,self.variable,self.jets_captured,method=method)
        self.sixb_position = get_sixb_position(self.jets_captured_index,self.sixb_jet_mask)
        self.sixb_captured_index = self.jets_captured_index[self.sixb_position]
        
        self.select_njets(njets)
        
    def sorted_jets(self,variable,njets=-1,method=max):
        new_selection = self.copy()
        new_selection.sort_jets(variable,njets,method)
        return new_selection
        
    def select_njets(self,njets):
        self.njets = njets
        self.jets_selected_index, self.jets_remaining_index = get_top_njet_index(self.branches,self.jets_captured_index,self.njets)
        
        self.jets_selected = get_jet_index_mask(self.branches,self.jets_selected_index)
        self.njets_selected = ak.sum(self.jets_selected,axis=-1)
        
        self.jets_remaining = get_jet_index_mask(self.branches,self.jets_remaining_index)
        self.njets_remaining = ak.sum(self.jets_remaining,axis=-1)
        
        self.sixb_selected_position = get_sixb_position(self.jets_selected_index,self.sixb_jet_mask)
        self.sixb_selected_index = self.jets_selected_index[self.sixb_selected_position]
        self.sixb_selected = get_jet_index_mask(self.branches,self.sixb_selected_index)
        self.nsixb_selected = ak.sum(self.sixb_selected,axis=-1)
        
        self.sixb_remaining_position = get_sixb_position(self.jets_remaining_index,self.sixb_jet_mask)
        self.sixb_remaining_index = self.jets_remaining_index[self.sixb_remaining_position]
        self.sixb_remaining = get_jet_index_mask(self.branches,self.sixb_remaining_index)
        self.nsixb_remaining = ak.sum(self.sixb_remaining,axis=-1)
        
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
        
        self.sixb_selected_position = get_sixb_position(self.jets_selected_index,self.sixb_jet_mask)
        self.sixb_selected_index = self.jets_selected_index[self.sixb_selected_position]
        
    def sorted_selected_jets(self,variable):
        new_selection = self.copy()
        new_selection.sort_selected_jets(variable)
        return new_selection
    
    def reco_X(self):
        fill_zero = lambda ary : ak.fill_none(ak.pad_none(ary,6,axis=-1,clip=1),0)
        
        jets = self.jets_selected_index
        jet_pt = fill_zero(self.branches["jet_ptRegressed"][jets])
        jet_eta = fill_zero(self.branches["jet_eta"][jets])
        jet_phi = fill_zero(self.branches["jet_phi"][jets])
        jet_m = fill_zero(self.branches["jet_m"][jets])

        ijet_p4 = [ vector.obj(pt=jet_pt[:,ijet],eta=jet_eta[:,ijet],phi=jet_phi[:,ijet],mass=jet_m[:,ijet]) for ijet in range(self.njets) ]
        X_reco = ijet_p4[0]+ijet_p4[1]+ijet_p4[2]+ijet_p4[3]+ijet_p4[4]+ijet_p4[5]
        return {"m":X_reco.mass,"pt":X_reco.pt,"eta":X_reco.eta,"phi":X_reco.phi}
        
    def score(self): 
        return SelectionScore(self)
    
    def merge(self):
        return MergedSelection(self)
    
    def copy(self):
        return CopySelection(self)
    
    def title(self,i=0):
        ignore = lambda tag : any( _ in tag for _ in ["baseline","preselection"] )
        if self.tag is None: return
        title = f"{self.njets} {self.tag}" if self.njets != -1 else f"all {self.tag}"
        if self.variable is not None: title = f"{title} / {self.variable.replace('jet_','')}"
        if self.include and self.include.tag and not ignore(self.include.tag): 
            title = f"{self.include.title(1)} & {title}"
            if i != 1: title = f"({title})" 
        if self.previous and self.previous.tag and not ignore(self.previous.tag): 
            title = f"{self.previous.title(2)} | {title}"
            
        if i != 0: return title
        
        tag_map = {}
        for tag in re.split('\s[&|]\s',title): 
            tag = re.sub('[\(\)]','',tag)
            if tag not in tag_map: tag_map[tag] = f"a{len(tag_map)}"
        tag_eq = title.replace(" & ","*").replace(" | ","+")#.replace(" / ","/")
        for tag,var in tag_map.items(): tag_eq = tag_eq.replace(tag,var)
        tag_eq_sim = str(sp.simplify(tag_eq)).replace(" ","")
        tag_sim = tag_eq_sim.replace("*"," & ").replace("+"," | ")#.replace("/"," / ")
        for tag,var in tag_map.items(): tag_sim = tag_sim.replace(var,tag)
        
        return tag_sim
    
    def __str__(self):
        return f"--- {self.title()} ---\n{self.score()}"
           
class SelectionScore:
    def __init__(self,selection):
        branches = selection.branches
        previous_nevents = selection.previous_nevents

        njets = selection.njets
        if njets < 0: njets = 6
        nsixb = min(6,njets)
        
        total_nsixb = ak.sum(branches["nfound_all"][selection.mask])
        
        total_min_nsixb = ak.sum(array_min(branches["nfound_all"][selection.mask],nsixb))
        total_remain = ak.sum(selection.nsixb_remaining[selection.mask])
        
        total_captured = ak.sum(selection.nsixb_captured[selection.mask])
        total_sig_captured = ak.sum(selection.nsixb_captured[selection.mask] == nsixb)
        
        total_selected = ak.sum(selection.nsixb_selected[selection.mask])
        total_sig_selected = ak.sum(selection.nsixb_selected[selection.mask] == nsixb)
        
        self.event_eff = selection.nevents/float(previous_nevents)
        
        self.event_avg_captured = total_captured/float(selection.nevents)
        self.event_per_captured = total_captured/float(total_nsixb)
        self.event_pur_captured = total_sig_captured/float(selection.nevents)
        self.event_total_captured = total_captured/float(total_nsixb)
        
        self.event_avg_selected = total_selected/float(selection.nevents)
        self.event_per_selected = total_selected/float(total_min_nsixb)
        self.event_pur_selected = total_sig_selected/float(selection.nevents)
        self.event_total_selected = total_selected/float(total_nsixb)
    def __str__(self):
        prompt_list = [
            "Event Efficiency:      {event_eff:0.2}",
#             "Event Captured Purity: {event_pur_captured:0.2f}",
            "Event Selected Purity: {event_pur_selected:0.2f}",
        ]
        prompt = '\n'.join(prompt_list)
        return prompt.format(**vars(self))
    def savetex(self,fname):
        prompt_list = [
            "Event Efficiency:      {event_eff:0.2}",
#             "Event Captured Purity: {event_pur_captured:0.2f}",
            "Event Selected Purity: {event_pur_selected:0.2f}",
        ]
        output = '\\\\ \n'.join(prompt_list).format(**vars(self))
        with open(f"{fname}.tex","w") as f: f.write(output)
            
class CopySelection(Selection):
    def __init__(self,selection):
        for key,value in vars(selection).items():
            setattr(self,key,value)
        
class MergedSelection(CopySelection): 
    def __init__(self,selection): 
        CopySelection.__init__(self,selection)

        previous = selection.previous
        while(previous is not None): 
            self.add(previous)
            self.previous_nevents = previous.previous_nevents
            previous = previous.previous
        self.nsixb_captured = ak.sum(self.sixb_captured,axis=-1)
        self.jets_captured_index = ak.concatenate((self.jets_selected_index,self.jets_remaining_index),axis=-1)
        self.sixb_position = get_sixb_position(self.jets_captured_index,self.sixb_jet_mask)
        self.sixb_selected_position = get_sixb_position(self.jets_selected_index,self.sixb_jet_mask)
        self.sixb_captured_index = self.jets_captured_index[self.sixb_position]
        self.previous = selection
        self.njets = self.total_njets
        self.tag = "merged"
        self.variable = None
        
    def add(self,selection):
        for key in ("jets_captured","sixb_captured"):
            setattr(self,key, getattr(self,key) | getattr(selection,key))
        for key in ("jets_selected_index","sixb_selected_index"):
            setattr(self,key, ak.concatenate((getattr(selection,key),getattr(self,key)),axis=-1))
        for key in ("njets_selected","nsixb_selected"):
            setattr(self,key, getattr(self,key) + getattr(selection,key))
        for key in ("jets_selected","sixb_selected"):
            setattr(self,key, getattr(self,key) | getattr(selection,key))
        
    def title(self): return f"{self.previous.title()} merged"
    
class SignalSelection(MergedSelection):
    signal_methods = {
        "xmass":xmass_selected_signal,
        "top":get_top_njet_index
    }
    def __init__(self,branches,previous,njets=6,mask=None,method="xmass"):
        MergedSelection.__init__(self,previous)
        
        self.variable = None
        
        self.previous = previous
        
        self.previous_index = self.jets_selected_index
        self.previous_njets = self.total_njets
        
        self.jets_order = self.jets_selected_index
        self.jets_captured = self.jets_selected
        self.njets_captured = ak.sum(self.jets_captured,axis=-1)
        
        self.sixb_captured = self.jets_captured & self.sixb_jet_mask
        self.nsixb_captured = ak.sum(self.sixb_captured,axis=-1)
        
        self.previous_events_mask = self.previous.mask
        self.previous_nevents = ak.sum(self.previous_events_mask)
        
        if mask is not None: self.mask = self.mask & mask
        
        self.select_signal(njets,method)
        
    def select_signal(self,njets=6,method="xmass"):
        self.tag = f"{method} selected"
        self.njets = min(6,njets) if njets != -1 else 6
        if method == "top": njets = self.njets
        
        self.jets_selected_index, self.jets_remaining_index = self.signal_methods[method](self.branches,self.previous_index,njets=njets)
    
        self.jets_selected = get_jet_index_mask(self.branches,self.jets_selected_index)
        self.njets_selected = ak.sum(self.jets_selected,axis=-1)
        
        self.jets_remaining = get_jet_index_mask(self.branches,self.jets_remaining_index)
        self.njets_remaining = ak.sum(self.jets_remaining,axis=-1)
        
        self.sixb_selected_position = get_sixb_position(self.jets_selected_index,self.sixb_jet_mask)
        self.sixb_selected_index = self.jets_selected_index[self.sixb_selected_position]
        self.sixb_selected = get_jet_index_mask(self.branches,self.sixb_selected_index)
        self.nsixb_selected = ak.sum(self.sixb_selected,axis=-1)
        
        self.sixb_remaining_position = get_sixb_position(self.jets_remaining_index,self.sixb_jet_mask)
        self.sixb_remaining_index = self.jets_remaining_index[self.sixb_remaining_position]
        self.sixb_remaining = get_jet_index_mask(self.branches,self.sixb_remaining_index)
        self.nsixb_remaining = ak.sum(self.sixb_remaining,axis=-1)
        
    def title(self): return Selection.title(self)
