
from datetime import date
import matplotlib.pyplot as plt
import numpy as np
import os
from .. import *

varinfo = {
    f"jet_m":     {"bins":np.linspace(0,60,50)      ,"xlabel":"Jet Mass"},
    f"jet_E":     {"bins":np.linspace(0,1000,50)     ,"xlabel":"Jet Energy"},
    f"jet_pt":    {"bins":np.linspace(0,1000,50)     ,"xlabel":"Jet Pt (GeV)"},
    f"jet_btag":  {"bins":np.linspace(0,1,50)       ,"xlabel":"Jet Btag"},
    f"jet_qgl":   {"bins":np.linspace(0,1,50)       ,"xlabel":"Jet QGL"},
    f"jet_min_dr":{"bins":np.linspace(0,3,50)       ,"xlabel":"Jet Min dR"},
    f"jet_eta":   {"bins":np.linspace(-3,3,50)      ,"xlabel":"Jet Eta"},
    f"jet_phi":   {"bins":np.linspace(-3.14,3.14,50),"xlabel":"Jet Phi"},
}

date_tag = date.today().strftime("%Y%m%d")

def save_scores(score,saveas):
    directory = f"plots/{date_tag}_plots/scores"
    if not os.path.isdir(directory): os.makedirs(directory)
    score.savetex(f"{directory}/{saveas}")
    
def save_fig(fig,directory,saveas):
    directory = f"plots/{date_tag}_plots/{directory}"
    if not os.path.isdir(directory): os.makedirs(directory)
    fig.savefig(f"{directory}/{saveas}.pdf",format="pdf")
    
class Study:
    def __init__(self,selection,title=None,saveas=None,print_score=True,subset="selected",mask=None,varlist=["jet_pt","jet_btag","jet_qgl","jet_eta"],autobin=False,**kwargs):
        if subset not in ["selected","passed","remaining","failed"]: raise ValueError(f"{subset} not available")
        if mask is not None: selection = selection.masked(mask)
        self.selection = selection
        self.scale = selection.nevents*selection.scale
        self.subset = subset
        self.saveas = saveas
        self.varinfo = { var:dict(**varinfo[var]) for var in varlist }
        
        if autobin: 
            for var in self.varinfo.values(): var["bins"] = None
        
        if title is None: title = selection.title()
        self.title = title
        print(f"--- {title} ---")
        
        score = selection.score()
        if print_score: print(score)
        if saveas: save_scores(score,saveas)
        
from .signal_study import *
