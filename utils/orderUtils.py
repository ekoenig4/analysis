from . import *
from .cutConfig import *

def btag_bias_pt_ordering(tree,baseline=None,tag="btag bias pt ordered"):
    if baseline is None: baseline = Selection(baseline)
    selection = Selection(tree,cuts=dict(btagcut=jet_btagWP[tightWP]),include=baseline)
    selection = Selection(tree,cuts=dict(btagcut=jet_btagWP[mediumWP]),previous=selection,include=baseline)
    selection = Selection(tree,cuts=dict(btagcut=jet_btagWP[looseWP]),previous=selection,include=baseline)
    selection = Selection(tree,cuts=dict(btagcut=jet_btagWP[nullWP]),previous=selection,include=baseline)
    return selection.merge(tag)

def jet_cut(tree,ordered=None):
    if ordered is None: ordered = Selection(tree)
    selection = Selection(tree,cuts=dict(njetcut=1,ptcut=60,btagcut=jet_btagWP[tightWP]),njets=1,include=ordered,tag="T60")
    selection = Selection(tree,cuts=dict(njetcut=1,ptcut=40,btagcut=jet_btagWP[tightWP]),njets=1 ,previous=selection,include=ordered,tag="T40")
    selection = Selection(tree,cuts=dict(njetcut=1,ptcut=40,btagcut=jet_btagWP[mediumWP]),njets=1,previous=selection,include=ordered,tag="M40")
    selection = Selection(tree,cuts=dict(njetcut=1,ptcut=20,btagcut=jet_btagWP[mediumWP]),njets=1,previous=selection,include=ordered,tag="M20")
    selection = Selection(tree,cuts=dict(njetcut=1,ptcut=20,btagcut=jet_btagWP[looseWP]),njets=1 ,previous=selection,include=ordered,tag="L20")
    selection = Selection(tree,cuts=dict(njetcut=1,ptcut=20,btagcut=jet_btagWP[looseWP]),njets=1 ,previous=selection,include=ordered,tag="L20")
    selection = Selection(tree,previous=selection,include=ordered,tag="remaining")
    return selection.merge("signal selection")
