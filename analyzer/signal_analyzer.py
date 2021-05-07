import ROOT as rt
import uproot as ut
from array import array
from argparse import ArgumentParser
from modules.kinematics import calcDeltaR

parser = ArgumentParser()
parser.add_argument("--file","-f")
parser.add_argument("--report","-r",type=int,default=10000)
parser.add_argument("--output","-o",default="output.root")

args = parser.parse_args()

tfile = rt.TFile(args.file,"update")
ttree = tfile.Get("sixBtree")

class EventInfo:
    def __init__(self,ttree):
        self.ttree = ttree
        self.variables = {
            "truH_pt":array('f',[0]*3),
            "truH_m":array('f',[0]*3),
            "truH_eta":array('f',[0]*3),
            "truH_phi":array('f',[0]*3),
            
            "truH_boost_v_mag":array('f',[0]*3),
            "truH_boost_v_pt":array('f',[0]*3),
            "truH_boost_v_eta":array('f',[0]*3),
            "truH_boost_v_phi":array('f',[0]*3),
            
            "nonH_pt":array('f',[0]*12),
            "nonH_m":array('f',[0]*12),
            "nonH_eta":array('f',[0]*12),
            "nonH_phi":array('f',[0]*12),
            
            "nonH_boost_v_mag":array('f',[0]*12),
            "nonH_boost_v_pt":array('f',[0]*12),
            "nonH_boost_v_eta":array('f',[0]*12),
            "nonH_boost_v_phi":array('f',[0]*12),

            "truH_bjet_dR":array('f',[0]*6),
            "nonH_bjet_dR":array('f',[0]*24),

            "truH_bjet_boost_pt":array('f',[0]*6),
            "truH_bjet_boost_m":array('f',[0]*6),
            "truH_bjet_boost_eta":array('f',[0]*6),
            "truH_bjet_boost_phi":array('f',[0]*6),
            "truH_bjet_boost_dR":array('f',[0]*6),
            
            "nonH_bjet_boost_pt":array('f',[0]*24),
            "nonH_bjet_boost_m":array('f',[0]*24),
            "nonH_bjet_boost_eta":array('f',[0]*24),
            "nonH_bjet_boost_phi":array('f',[0]*24),
            "nonH_bjet_boost_dR":array('f',[0]*24),
        }
        self.branches = { key:ttree.Branch(key,value,"%s[%i]/F"%(key,len(value))) for key,value in self.variables.items() }
    def fill(self):
        for branch in self.branches.values(): branch.Fill()

info = EventInfo(ttree)

class BJet:
    def __init__(self,event,tag):
        self.m = getattr(event,f"{tag}_recojet_m")
        self.pt = getattr(event,f"{tag}_recojet_ptRegressed")
        self.eta = getattr(event,f"{tag}_recojet_eta")
        self.phi = getattr(event,f"{tag}_recojet_phi")
    def getLorentz(self):
        lorentz = rt.TLorentzVector()
        lorentz.SetPtEtaPhiM(self.pt,self.eta,self.phi,self.m)
        return lorentz
    def getDR(self,bjet):
        return calcDeltaR(self.eta,bjet.eta,self.phi,bjet.phi)
class BoostedBJet(BJet):
    def __init__(self,bjet,boost):
        lorentz = bjet.getLorentz()
        lorentz.Boost(-boost)
        self.m = lorentz.M()
        self.pt = lorentz.Pt()
        self.eta = lorentz.Eta()
        self.phi = lorentz.Phi()
class Higgs:
    def __init__(self,b1,b2):
        lorentz = b1.getLorentz() + b2.getLorentz()
        self.dr1 = b1.getDR(b2)
        self.dr2 = b2.getDR(b1)
        self.m = lorentz.M()
        self.pt = lorentz.Pt()
        self.eta = lorentz.Eta()
        self.phi = lorentz.Phi()
    def getLorentz(self):
        lorentz = rt.TLorentzVector()
        lorentz.SetPtEtaPhiM(self.pt,self.eta,self.phi,self.m)
        return lorentz
    def getBoost(self): return self.getLorentz().BoostVector()
def process(event):
    tags = ["HX","HY1","HY2"]

    bjetlist = []

    for h in tags:
        for b in ("b1","b2"):
            bjetlist.append( BJet(event,f"{h}_{b}"))
    higgslist = { "tru":[],"non":[] }
    boostlist = { "tru":[],"non":[] }
    boosted_jetlist = { "tru":[],"non":[] }
    boosted_higgslist = {"tru":[],"non":[]}
    for i1 in range(6):
        b1 = bjetlist[i1]
        for i2 in range(i1+1,6):
            
            b2 = bjetlist[i2]
            higgs = Higgs(b1,b2)
            boost = higgs.getBoost()
            boosted_b1 = BoostedBJet(b1,boost)
            boosted_b2 = BoostedBJet(b2,boost)
            boosted_higgs = Higgs(boosted_b1,boosted_b2)


            key = "tru" if ( i2-i1 == 1 and i1%2 == 0 ) else "non"
            higgslist[key].append(higgs)
            boostlist[key].append(boost)
            boosted_jetlist[key].append( (boosted_b1,boosted_b2) )
            boosted_higgslist[key].append(boosted_higgs)
            
    for key in ("tru","non"):
        for i,(higgs,boost,(boosted_b1,boosted_b2),boosted_higgs) in enumerate( zip(higgslist[key],boostlist[key],boosted_jetlist[key],boosted_higgslist[key]) ):
            info.variables[f"{key}H_pt"][i] = higgs.pt
            info.variables[f"{key}H_m"][i] = higgs.m
            info.variables[f"{key}H_eta"][i] = higgs.eta
            info.variables[f"{key}H_phi"][i] = higgs.phi
            info.variables[f"{key}H_bjet_dR"][2*i] = higgs.dr1
            info.variables[f"{key}H_bjet_dR"][2*i+1] = higgs.dr2
            
            info.variables[f"{key}H_boost_v_pt"][i] = boost.Pt()
            info.variables[f"{key}H_boost_v_mag"][i] = boost.Mag()
            info.variables[f"{key}H_boost_v_eta"][i] = boost.Eta()
            info.variables[f"{key}H_boost_v_phi"][i] = boost.Phi()
            
            info.variables[f"{key}H_bjet_boost_pt"][2*i] = boosted_b1.pt
            info.variables[f"{key}H_bjet_boost_m"][2*i] = boosted_b1.m
            info.variables[f"{key}H_bjet_boost_eta"][2*i] = boosted_b1.eta
            info.variables[f"{key}H_bjet_boost_phi"][2*i] = boosted_b1.phi
            info.variables[f"{key}H_bjet_boost_dR"][2*i] = boosted_higgs.dr1
            
            info.variables[f"{key}H_bjet_boost_pt"][2*i+1] = boosted_b2.pt
            info.variables[f"{key}H_bjet_boost_m"][2*i+1] = boosted_b2.m
            info.variables[f"{key}H_bjet_boost_eta"][2*i+1] = boosted_b2.eta
            info.variables[f"{key}H_bjet_boost_phi"][2*i+1] = boosted_b2.phi
            info.variables[f"{key}H_bjet_boost_dR"][2*i+1] = boosted_higgs.dr2
    info.fill()

    
nevents = ttree.GetEntriesFast()
for i,event in enumerate(ttree):
    if (i%args.report == 0): print(f"Processing {i} of {nevents}")
    process(event)
ttree.Write("",rt.TObject.kOverwrite)
