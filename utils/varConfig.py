# from attrdict import AttrDict
import numpy as np

class AttrDict:
    def __init__(self, *args, **kwargs):
        self.__dict__.update(*args, **kwargs)
    # def __getattr__(self, key):
    #     if key == '__dict__': return super().__getattr__(self, key)

    #     print('getting attr')
    #     value = self.__dict__[key]
    #     if isinstance(value, dict): return AttrDict(**value)
    #     return value

    def __setattr__(self, key, value):
        if isinstance(value, dict): value = AttrDict(value)
        self.__dict__[key] = value

    def __getitem__(self, key): return getattr(self, key)
    def __setitem__(self, key, value): setattr(self, key, value)

    def update(self, *args, **kwargs):
        values = dict()
        if isinstance(args, dict):
            values.update(args)
        else:
            for d in args:
                values.update(d)
        values.update(**kwargs)

        for key,value in values.items():
            setattr(self, key, value)
    def get(self, key, default=None): return self.__dict__.get(key, default)
    def clear(self): 
        keys = list(self.__dict__.keys())
        for key in keys:
            if key.startswith("__") and key.endswith("__"): continue
            del self.__dict__[key]

    def __contains__(self, key): return key in self.__dict__
    def __iter__(self): return iter(self.__dict__)
    def __len__(self): return len(self.__dict__)

    def __str__(self): return str(self.__dict__)
    def __repr__(self): return repr(self.__dict__)

class VarInfo(AttrDict):
    def find(self, var):
        if not isinstance(var, str):
            return AttrDict(
                xlabel=getattr(var, 'xlabel', getattr(var, '__name__', str(var))),
                bins=getattr(var, 'bins', None),
            )

        if var in self:
            return self[var]
        end_pattern = next((self[name]
                           for name in self if var.endswith(name)), None)
        if end_pattern:
            return end_pattern
        start_pattern = next(
            (self[name] for name in self if var.startswith(name)), None)
        if start_pattern:
            return start_pattern
        any_pattern = next((self[name] for name in self if var in name), None)
        if any_pattern:
            return any_pattern


varinfo = VarInfo()

# varinfo.jet_m = dict(bins=np.linspace(0, 60, 30), xlabel="Jet Mass")
# varinfo.jet_E = dict(bins=np.linspace(0, 1000, 30), xlabel="Jet Energy")
# varinfo.jet_pt = dict(bins=np.linspace(0, 1000, 30), xlabel="Jet Pt (GeV)")
# varinfo.jet_btag = dict(bins=np.linspace(0, 1, 30), xlabel="Jet Btag")
# varinfo.jet_qgl = dict(bins=np.linspace(0, 1, 30), xlabel="Jet QGL")
# varinfo.jet_min_dr = dict(bins=np.linspace(0, 3, 30), xlabel="Jet Min dR")
# varinfo.jet_eta = dict(bins=np.linspace(-3, 3, 30), xlabel="Jet Eta")
# varinfo.jet_phi = dict(bins=np.linspace(-3.14, 3.14, 30), xlabel="Jet Phi")
# varinfo.n_jet = dict(bins=np.arange(12), xlabel="N Jets")
# varinfo.higgs_m = dict(bins=np.linspace(0, 300, 30), xlabel="DiJet Mass")
# varinfo.higgs_E = dict(bins=np.linspace(0, 500, 30), xlabel="DiJet Energy")
# varinfo.higgs_pt = dict(bins=np.linspace(0, 500, 30), xlabel="DiJet Pt (GeV)")
# varinfo.higgs_eta = dict(bins=np.linspace(-3, 3, 30), xlabel="DiJet Eta")
# varinfo.higgs_phi = dict(bins=np.linspace(-3.14, 3.14, 30), xlabel="DiJet Phi")
# varinfo.dijet_m = dict(bins=np.linspace(0, 300, 30), xlabel="DiJet Mass")
# varinfo.dijet_E = dict(bins=np.linspace(0, 500, 30), xlabel="DiJet Energy")
# varinfo.dijet_pt = dict(bins=np.linspace(0, 500, 30), xlabel="DiJet Pt (GeV)")
# varinfo.dijet_eta = dict(bins=np.linspace(-3, 3, 30), xlabel="DiJet Eta")
# varinfo.dijet_phi = dict(bins=np.linspace(-3.14, 3.14, 30), xlabel="DiJet Phi")
# varinfo.n_higgs = dict(bins=np.arange(12), xlabel="N DiJets")
# varinfo.jet_btagsum = dict(bins=np.linspace(2, 6, 30), xlabel="6 Jet Btag Sum")
# varinfo.event_y23 = dict(xlabel="Event y23", bins=np.linspace(0, 0.25, 30))
# varinfo.M_eig_w1 = dict(xlabel="Momentum Tensor W1", bins=np.linspace(0, 1, 30))
# varinfo.M_eig_w2 = dict(xlabel="Momentum Tensor W2", bins=np.linspace(0, 1, 30))
# varinfo.M_eig_w3 = dict(xlabel="Momentum Tensor W3", bins=np.linspace(0, 1, 30))
varinfo.event_S = dict(xlabel="Event S", bins=np.linspace(0, 1, 30))
varinfo.event_St = dict(xlabel="Event S_T", bins=np.linspace(0, 1, 30))
varinfo.event_F = dict(xlabel="Event W2/W1", bins=np.linspace(0, 1, 30))
varinfo.event_A = dict(xlabel="Event A", bins=np.linspace(0, 0.5, 30))
varinfo.event_AL = dict(xlabel="Event A_L", bins=np.linspace(-1, 1, 30))
varinfo.thrust_phi = dict(xlabel="T_T Phi", bins=np.linspace(-3.14, 3.14, 30))
varinfo.event_Tt = dict(xlabel="1 - T_T", bins=np.linspace(0, 1/3, 30))
varinfo.event_Tm = dict(xlabel="T_m", bins=np.linspace(0, 2/3, 30))
varinfo.b_6j_score = dict(xlabel="6 Jet Classifier Score", bins=np.linspace(0, 1, 30))
varinfo.b_3d_score = dict(xlabel="3 Higgs Classifier Score",bins=np.linspace(0, 1, 30))
varinfo.b_2j_score = dict(xlabel="2 Jet Classifier Score", bins=np.linspace(0, 1, 30))

varinfo.X_m = dict(bins=np.linspace(500,2000,30))
varinfo.Y1_m = dict(bins=np.linspace(200,1000,30))
varinfo.Y2_m = dict(bins=np.linspace(200,1000,30))
varinfo.H1Y1_m = dict(bins=np.linspace(0,250,30))
varinfo.H2Y1_m = dict(bins=np.linspace(0,250,30))
varinfo.H1Y2_m = dict(bins=np.linspace(0,250,30))
varinfo.H2Y2_m = dict(bins=np.linspace(0,250,30))
