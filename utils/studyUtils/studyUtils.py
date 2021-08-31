from . import *

varinfo = {
    f"jet_m":     {"bins":np.linspace(0,60,30)      ,"xlabel":"Jet Mass"},
    f"jet_E":     {"bins":np.linspace(0,1000,30)     ,"xlabel":"Jet Energy"},
    f"jet_pt":    {"bins":np.linspace(0,1000,30)     ,"xlabel":"Jet Pt (GeV)"},
    f"jet_btag":  {"bins":np.linspace(0,1,30)       ,"xlabel":"Jet Btag"},
    f"jet_qgl":   {"bins":np.linspace(0,1,30)       ,"xlabel":"Jet QGL"},
    f"jet_min_dr":{"bins":np.linspace(0,3,30)       ,"xlabel":"Jet Min dR"},
    f"jet_eta":   {"bins":np.linspace(-3,3,30)      ,"xlabel":"Jet Eta"},
    f"jet_phi":   {"bins":np.linspace(-3.14,3.14,30),"xlabel":"Jet Phi"},
    f"n_jet":     {"bins":range(12)                 ,"xlabel":"N Jets"},
    f"higgs_m":   {"bins":np.linspace(0,300,30)      ,"xlabel":"DiJet Mass"},
    f"higgs_E":   {"bins":np.linspace(0,1500,30)     ,"xlabel":"DiJet Energy"},
    f"higgs_pt":  {"bins":np.linspace(0,1500,30)     ,"xlabel":"DiJet Pt (GeV)"},
    f"higgs_eta": {"bins":np.linspace(-3,3,30)      ,"xlabel":"DiJet Eta"},
    f"higgs_phi": {"bins":np.linspace(-3.14,3.14,30),"xlabel":"DiJet Phi"},
    f"n_higgs":   {"bins":range(12)                 ,"xlabel":"N DiJets"},
    f"jet6_btagsum":{"bins":np.linspace(2,6,30)     ,"xlabel":"6 Jet Btag Sum"}
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
    
def save_fig(fig,directory,saveas,base=GIT_WD):
    directory = f"{base}/plots/{date_tag}_plots/{directory}"
    if not os.path.isdir(directory): os.makedirs(directory)
    fig.savefig(f"{directory}/{saveas}.pdf",format="pdf")
    
class Study:
    def __init__(self,selections,labels=None,density=0,log=0,ratio=0,stacked=0,lumikey=2018,sumw2=True,title=None,saveas=None,masks=None,varlist=varinfo.keys(),**kwargs):
        if type(selections) == tuple: selections = list(selections)
        if type(selections) != list: selections = [selections]
        
        self.selections = selections
        self.masks = masks

        self.attrs = dict(
            labels = labels if labels else [ selection.tag for selection in selections ],
            is_datas = [ selection.is_data for selection in selections ],
            is_signals = [ selection.is_signal for selection in selections ],
            s_colors = [ selection.color for selection in selections ],
            
            density = density,
            log = log,
            ratio = ratio,
            stacked = stacked,
            lumikey = lumikey,
            sumw2 = sumw2,
            **kwargs,
        )
        
        
        self.title = title
        self.saveas = saveas
        self.varinfo = {key:varinfo[key] for key in varlist}

    def get(self,key):
        items = [ selection[key] for selection in self.selections ]
        if self.masks is not None:
            items = [ item[mask] for item,mask in zip(items,self.masks) ]
        return items
