from . import *

from attrdict import AttrDict

class VarInfo(AttrDict):
    def find(self,var):
        if var in self: return self[var]
        end_pattern = next( (info for name,info in self.items() if name.endswith(var)),None )
        if end_pattern: return end_pattern
        start_pattern = next( (info for name,info in self.items() if name.startswith(var)),None )
        if start_pattern: return start_pattern
        any_pattern = next( (info for name,info in self.items() if var in name),None )
        if any_pattern: return any_pattern

varinfo = VarInfo({
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
    f"jet_btagsum":{"bins":np.linspace(2,6,30)     ,"xlabel":"6 Jet Btag Sum"},
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
})
