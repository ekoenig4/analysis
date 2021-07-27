
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
    f"n_jet":     {"bins":range(12)                 ,"xlabel":"N Jets"},
}

shapeinfo = {
    "event_y23":dict(xlabel="Event y23",bins=np.linspace(0,0.25,30)),
    "M_eig_w1":dict(xlabel="Momentum Tensor W1",bins=np.linspace(0,1,30)),
    "M_eig_w2":dict(xlabel="Momentum Tensor W2",bins=np.linspace(0,1,30)),
    "M_eig_w3":dict(xlabel="Momentum Tensor W3",bins=np.linspace(0,1,30)),
    "event_S":dict(xlabel="Event S",bins=np.linspace(0,1,30)),
    "event_St":dict(xlabel="Event S_T",bins=np.linspace(0,1,30)),
    "event_F":dict(xlabel="Event W2/W1",bins=np.linspace(0,1,30)),
    "event_A":dict(xlabel="Event A",bins=np.linspace(0,0.5,30)),
    "event_AL":dict(xlabel="Event A_L",bins=np.linspace(-1,1,30)),
    "thrust_phi":dict(xlabel="T_T Phi",bins=np.linspace(-3.14,3.14,30)),
    "event_Tt":dict(xlabel="1 - T_T",bins=np.linspace(0,1/3,30)),
    "event_Tm":dict(xlabel="T_m",bins=np.linspace(0,2/3,30)),
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
        if type(selections) == tuple: selections = list(selections)
        if type(selections) != list: selections = [selections]
        if mask is not None: selections = [ selection.masked(mask) for selection in selections ]
        
        self.selections = selections
        self.labels = labels if labels else [ selection.tag for selection in selections ]
        self.title = selections[0].title() if title is None and len(selections) == 1 else title
        self.density = density
        self.log = log
        self.lumikey = lumikey
        self.saveas = saveas
        self.varinfo = dict(**varinfo)
        
        
from .signal_studies import signal_study
from .studies import study
