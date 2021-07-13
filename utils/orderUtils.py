from . import *
from .cutConfig import *

def btag_bias_pt_ordering(branches,baseline,tag="btag bias pt ordered"):
    selection = Selection(branches,cuts=dict(btagcut=jet_btagWP[tightWP]),include=baseline)
    selection = Selection(branches,cuts=dict(btagcut=jet_btagWP[mediumWP]),previous=selection,include=baseline)
    selection = Selection(branches,cuts=dict(btagcut=jet_btagWP[looseWP]),previous=selection,include=baseline)
    selection = Selection(branches,cuts=dict(btagcut=jet_btagWP[nullWP]),previous=selection,include=baseline)
    return selection.merge(tag)

def signal_ordered(branches,ordered):
    selection = Selection(branches,cuts=dict(njetcut=1,ptcut=60,btagcut=jet_btagWP[tightWP]),njets=1,include=ordered,tag="T60")
    selection = Selection(branches,cuts=dict(njetcut=1,ptcut=40,btagcut=jet_btagWP[tightWP]),njets=1 ,previous=selection,include=ordered,tag="T40")
    selection = Selection(branches,cuts=dict(njetcut=1,ptcut=40,btagcut=jet_btagWP[mediumWP]),njets=1,previous=selection,include=ordered,tag="M40")
    selection = Selection(branches,cuts=dict(njetcut=1,ptcut=20,btagcut=jet_btagWP[mediumWP]),njets=1,previous=selection,include=ordered,tag="M20")
    selection = Selection(branches,cuts=dict(njetcut=1,ptcut=20,btagcut=jet_btagWP[looseWP]),njets=1 ,previous=selection,include=ordered,tag="L20")
    selection = Selection(branches,cuts=dict(njetcut=1,ptcut=20,btagcut=jet_btagWP[looseWP]),njets=1 ,previous=selection,include=ordered,tag="L20")
    selection = Selection(branches,previous=selection,include=ordered,tag="remaining")
    return selection.merge("signal selection")
