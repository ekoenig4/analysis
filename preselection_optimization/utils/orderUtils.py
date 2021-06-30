from . import *

def btag_bias_pt_ordering(branches,baseline,tag="btag bias pt ordered"):
    selection = Selection(branches,cuts=dict(btagcut=jet_btagWP[tightWP]),include=baseline)
    selection = Selection(branches,cuts=dict(btagcut=jet_btagWP[mediumWP]),previous=selection,include=baseline)
    selection = Selection(branches,cuts=dict(btagcut=jet_btagWP[looseWP]),previous=selection,include=baseline)
    selection = Selection(branches,cuts=dict(btagcut=jet_btagWP[nullWP]),previous=selection,include=baseline)
    return selection.merge(tag)