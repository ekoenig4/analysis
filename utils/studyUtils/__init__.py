
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
    def __init__(self,selections,labels=None,density=0,log=0,lumikey=2018,title=None,saveas=None,mask=None,**kwargs):
        if type(selections) != list: selections = [selections]
        if mask is not None: selections = [ selection.masked(mask) for selection in selections ]
        
        self.selections = selections
        self.labels = labels if labels else [ selection.tag for selection in selections ]
        self.title = selections[0].title() if title is None and len(selections) == 1 else title
        self.denisty = density
        self.log = log
        self.lumikey = lumikey
        self.saveas = saveas
        self.varinfo = dict(**varinfo)
        
        
from .signal_studies import signal_study
from .studies import study
