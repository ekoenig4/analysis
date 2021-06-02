from . import *

class Branches:
    def __init__(self,ttree):
        self.ttree = ttree
        self.nevents = ak.size( self["Run"] )

        self.sixb_found_mask = self["nfound_all"] == 6
        self.nsignal = ak.sum( self.sixb_found_mask )
        self.sixb_jet_mask = get_jet_index_mask(self,self["signal_bjet_index"])
        self.set_ptbtag()
    def __getitem__(self,key): 
        if key == "jet_ptbtag": return self.jet_ptbtag
        return self.ttree[key].array()
    def set_ptbtag(self,alpha=1,beta=1,ptscale=200.):
        self.jet_ptbtag = (alpha*self["jet_ptRegressed"]/ptscale+1)*(alpha*self["jet_btag"]+1)
    
class Selection:
    def __init__(self,branches,scheme={},previous=None,exclude=False,variable=None,njets=None,tag=""):
        self.tag = tag
        self.branches = branches
        self.scheme = scheme
        self.exclude = exclude
        
        self.previous = previous
        self.begin = None
        if previous: self.begin = previous.begin if previous.begin else previous
        
        self.exclude_jet_mask = None if previous is None else previous.jets_selected
        self.exclude_events_mask = None if previous is None else previous.mask
        
        self.sixb_jet_mask = branches.sixb_jet_mask
        
        self.previous_njets = 0 if previous is None else previous.total_njets
        
        self.mask, self.jet_mask = std_preselection(self.branches,exclude_events_mask=self.exclude_events_mask,
                                                    exclude_jet_mask=self.exclude_jet_mask,exclude=self.exclude,**self.scheme)
        self.sixb_captured = self.jet_mask & self.sixb_jet_mask
        self.nsixb_captured = ak.sum(self.sixb_captured,axis=-1)
        
        self.nevents = ak.sum(self.mask)
        self.nsignal = ak.sum(self.mask & branches.sixb_found_mask)
        
        if variable is not None:
            self.sort_jets(variable)
            if njets is not None:
                self.select_njets(njets)
        
    def sort_jets(self,variable,njets=-1):
        self.variable = variable
        self.jets_ordered = sort_jet_index(self.branches,self.variable,self.jet_mask)
        self.sixb_position = get_sixb_position(self.jets_ordered,self.sixb_jet_mask)
        self.sixb_ordered = self.jets_ordered[self.sixb_position]
        
        self.select_njets(njets)
        
    def sorted_jets(self,variable,njets=-1):
        new_selection = self.copy()
        new_selection.sort_jets(variable,njets)
        return new_selection
        
    def select_njets(self,njets):
        self.njets = njets
        self.jets_selected_index, self.jets_remaining_index = get_top_njet_index(self.branches,self.jets_ordered,self.njets)
        
        self.jets_selected = get_jet_index_mask(self.branches,self.jets_selected_index)
        self.njets_selected = ak.sum(self.jets_selected,axis=-1)
        
        self.jets_remaining = get_jet_index_mask(self.branches,self.jets_remaining_index)
        self.njets_remaining = ak.sum(self.jets_remaining,axis=-1)
        
        sixb_selected_position = get_sixb_position(self.jets_selected_index,self.sixb_jet_mask)
        self.sixb_selected_index = self.jets_selected_index[sixb_selected_position]
        self.sixb_selected = get_jet_index_mask(self.branches,self.sixb_selected_index)
        self.nsixb_selected = ak.sum(self.sixb_selected,axis=-1)
        
        sixb_remaining_position = get_sixb_position(self.jets_remaining_index,self.sixb_jet_mask)
        self.sixb_remaining_index = self.jets_remaining_index[sixb_remaining_position]
        self.sixb_remaining = get_jet_index_mask(self.branches,self.sixb_remaining_index)
        self.nsixb_remaining = ak.sum(self.sixb_remaining,axis=-1)
        
        self.total_njets = self.previous_njets + self.njets
        
    def selected_njets(self,njets):
        new_selection = self.copy()
        new_selection.select_njets(njets)
        return new_selection
        
    def score(self): 
        return SelectionScore(self)
    
    def merge(self):
        return MergedSelection(self)
    
    def copy(self):
        return CopySelection(self)
    
    def title(self):
        previous = f"{self.previous.title()} & " if self.previous else ""
        njets = f"{self.njets} " if self.njets > 0 else "all "
        return f"{previous}{njets}{self.tag}"
    
    def __str__(self):
        return f"--- {self.title()} ---\n{self.score()}"
        
           
class SelectionScore:
    def __init__(self,selection):
        branches = selection.branches
        
        previous_nevents = selection.previous.nevents if selection.previous else branches.nevents
        previous_nsignal = selection.previous.nsignal if selection.previous else branches.nsignal

        njets = selection.njets
        if njets < 0: njets = 6
        nsixb = min(6,njets)

        total_nsixb = ak.sum(branches["nfound_all"][selection.mask])
        signal_nsixb = ak.sum(branches["nfound_all"][selection.mask & branches.sixb_found_mask])
        
        total_min_nsixb = ak.sum(array_min(branches["nfound_all"][selection.mask],nsixb))
        total_remain = ak.sum(selection.nsixb_remaining[selection.mask])
        signal_remain = ak.sum(selection.nsixb_remaining[selection.mask & branches.sixb_found_mask])
        total_selected = ak.sum(selection.nsixb_selected[selection.mask])
        signal_selected = ak.sum(selection.nsixb_selected[selection.mask & branches.sixb_found_mask])
        total_captured = ak.sum(selection.nsixb_captured[selection.mask])
        signal_captured = ak.sum(selection.nsixb_captured[selection.mask & branches.sixb_found_mask])
        avg_position = ak.mean(selection.sixb_position,axis=-1)
        
        all_sixb_selected = ak.sum(selection.nsixb_selected[selection.mask] == nsixb)
        
        self.event_eff = selection.nevents/float(previous_nevents)
        self.event_avg_selected = total_selected/float(selection.nevents)
        self.event_per_selected = total_selected/float(total_min_nsixb)
        self.event_total_selected = total_selected/float(total_nsixb)
        self.event_avg_captured = total_captured/float(selection.nevents)
        self.event_per_captured = total_captured/float(total_nsixb)
        self.event_avg_position = ak.mean(avg_position[selection.mask])
        self.event_purity = all_sixb_selected/float(selection.nevents)

        self.signal_eff = selection.nsignal/float(previous_nsignal)
        self.signal_avg_selected = signal_selected/float(selection.nsignal)
        self.signal_per_selected = signal_selected/float(nsixb*selection.nsignal)
        self.signal_total_selected = signal_selected/float(signal_nsixb)
        self.signal_avg_captured = signal_captured/float(selection.nsignal)
        self.signal_per_captured = signal_captured/float(signal_nsixb)
        self.signal_avg_position = ak.mean(avg_position[selection.mask & branches.sixb_found_mask])
        self.signal_purity = all_sixb_selected/float(selection.nsignal) 
        
        self.higgs_purity = selection.nsignal/float(selection.nevents)
    def __str__(self):
        prompt_list = [
            "Event  Efficiency:       {event_eff:0.2}",
            "Signal Efficiency:       {signal_eff:0.2f}",
            "Event  Selected Purity:  {event_purity:0.2f}",
            "Signal Selected Purity:  {signal_purity:0.2f}",
            "Event  Total Selected:   {event_total_selected:0.2%}",
            "Signal Total Selected:   {signal_total_selected:0.2%}",
            "Event  Avg Selected:     {event_avg_selected:0.2f} -> {event_per_selected:0.2%}",
            "Signal Avg Selected:     {signal_avg_selected:0.2f} -> {signal_per_selected:0.2%}",
            "Event  Avg Captured:     {event_avg_captured:0.2f} -> {event_per_captured:0.2%}",
            "Signal Avg Captured:     {signal_avg_captured:0.2f} -> {signal_per_captured:0.2%}",
        ]
        prompt = '\n'.join(prompt_list)
        return prompt.format(**vars(self))
    def savetex(self,fname):
        prompt_list = [
            "Event  Efficiency:       {event_eff:0.2}",
#             "Signal Efficiency:       {signal_eff:0.2f}",
            "Event  Selected Purity:  {event_purity:0.2f}",
#             "Signal Selected Purity:  {signal_purity:0.2f}",
#             "Event  Total Selected: {event_total_selected:0.2f}",
#             "Signal Total Selected: {signal_total_selected:0.2f}",
#             "Event  Avg Selected:   {event_avg_selected:0.2f}",
#             "Signal Avg Selected:   {signal_avg_selected:0.2f}",
#             "Event  Avg Captured:   {event_avg_captured:0.2f}",
#             "Signal Avg Captured:   {signal_avg_captured:0.2f}",
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
        self.previous = self.begin

        previous = selection.previous
        exclude = selection.exclude
        while(previous is not None): 
            self.add(previous,exclude)
            exclude = previous.exclude
            previous = previous.previous
        self.nsixb_captured = ak.sum(self.sixb_captured,axis=-1)
        self.jets_ordered = ak.concatenate((self.jets_selected_index,self.jets_remaining_index),axis=-1)
        self.sixb_position = get_sixb_position(self.jets_ordered,self.sixb_jet_mask)
        self.sixb_ordered = self.jets_ordered[self.sixb_position]
        
    def add(self,selection,exclude):
        if selection == self.begin: self.tag = f"({self.tag})"
        self.tag = f"{selection.tag} & {self.tag}"
        for key in ("jet_mask","sixb_captured"):
            setattr(self,key, getattr(self,key) | getattr(selection,key))
        if not exclude: return 
        
        for key in ("jets_selected_index","sixb_selected_index"):
            setattr(self,key, ak.concatenate((getattr(selection,key),getattr(self,key)),axis=-1))
        for key in ("njets","njets_selected","nsixb_selected"):
            setattr(self,key, getattr(self,key) + getattr(selection,key))
        for key in ("jets_selected","sixb_selected"):
            setattr(self,key, getattr(self,key) | getattr(selection,key))
            
    def title(self): return self.tag
        