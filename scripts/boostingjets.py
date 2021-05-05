import ROOT
import uproot3_methods

def PrintProperties(p4,object,frame):
    print("[INFO] Properties of %s in %s frame"%(object,frame))
    print("mass    :",p4.M())
    print("energy  :",p4.E())
    print("p       :",p4.P())
    print("pT      :",p4.Pt())
    print("pz      :",p4.Pz())
    print("phi     :",p4.Phi())
    print("eta     :",p4.Eta())
    print("theta   :",p4.Theta())
    print("costheta:",p4.CosTheta())
    
#Prepare your jets in lab frame
b1 = ROOT.TLorentzVector(0,0,0,0)
b1.SetPtEtaPhiM(40.0,2.0,1.2,5.0)
b2 = ROOT.TLorentzVector(0,0,0,0)
b2.SetPtEtaPhiM(30.0,1.2,0.7,8.0)

# b1 = uproot3_methods.TLorentzVectorArray.from_ptetaphim(40.0,2.0,1.2,5.0)
# b2 = uproot3_methods.TLorentzVectorArray.from_ptetaphim(30.0,1.2,0.7,8.0)

#Compute the boost vector associated to the jet pair
vboost = (b1+b2).BoostVector()
print(vboost)

#Prepare jets to be boosted to CM frame
b1_cm = ROOT.TLorentzVector(0,0,0,0)
b1_cm.SetPtEtaPhiM(40.0,2.0,1.2,5.0)
b2_cm = ROOT.TLorentzVector(0,0,0,0)
b2_cm.SetPtEtaPhiM(30.0,1.2,0.7,8.0)

#Boost jets to CM frame
b1_cm.Boost(-vboost)
b2_cm.Boost(-vboost)

#Print properties
PrintProperties(b1,"b1","lab")
PrintProperties(b2,"b2","lab")
PrintProperties(b1_cm,"b1","CM")
PrintProperties(b2_cm,"b2","CM")