"""
Utilities for processes ntuples from the Run 2 bbbbAnalysis code
"""

bbbb_variable_map = {
    'ak4_h1b1_bRegRes' : 'H1_b1_bRegRes',
    'ak4_h1b1_bdisc' : 'H1_b1_deepJet',
    'ak4_h1b1_btag_L' : '(H1_b1_deepJet > 0.0490)',
    'ak4_h1b1_btag_M' : 'H1_b1_deepJet > 0.2783',
    'ak4_h1b1_btag_T' : 'H1_b1_deepJet > 0.7100',
    'ak4_h1b1_eta' : 'H1_b1_eta',
    'ak4_h1b1_hflav' : 'H1_b1_hadronFlavour',
    'ak4_h1b1_mass' : 'H1_b1_m',
    'ak4_h1b1_phi' : 'H1_b1_phi',
    'ak4_h1b1_pt' : 'H1_b1_pt',
    'ak4_h1b1_regmass' : 'H1_b1_m',
    'ak4_h1b1_regpt' : 'H1_b1_ptRegressed',

    'ak4_h1b2_bRegRes' : 'H1_b2_bRegRes',
    'ak4_h1b2_bdisc' : 'H1_b2_deepJet',
    'ak4_h1b2_btag_L' : 'H1_b2_deepJet > 0.0490',
    'ak4_h1b2_btag_M' : 'H1_b2_deepJet > 0.2783',
    'ak4_h1b2_btag_T' : 'H1_b2_deepJet > 0.7100',
    'ak4_h1b2_eta' : 'H1_b2_eta',
    'ak4_h1b2_hflav' : 'H1_b2_hadronFlavour',
    'ak4_h1b2_mass' : 'H1_b2_m',
    'ak4_h1b2_phi' : 'H1_b2_phi',
    'ak4_h1b2_pt' : 'H1_b2_pt',
    'ak4_h1b2_regmass' : 'H1_b2_m',
    'ak4_h1b2_regpt' : 'H1_b2_ptRegressed',

    'ak4_h2b1_bRegRes' : 'H2_b1_bRegRes',
    'ak4_h2b1_bdisc' : 'H2_b1_deepJet',
    'ak4_h2b1_btag_L' : 'H2_b1_deepJet > 0.0490',
    'ak4_h2b1_btag_M' : 'H2_b1_deepJet > 0.2783',
    'ak4_h2b1_btag_T' : 'H2_b1_deepJet > 0.7100',
    'ak4_h2b1_eta' : 'H2_b1_eta',
    'ak4_h2b1_hflav' : 'H2_b1_hadronFlavour',
    'ak4_h2b1_mass' : 'H2_b1_m',
    'ak4_h2b1_phi' : 'H2_b1_phi',
    'ak4_h2b1_pt' : 'H2_b1_pt',
    'ak4_h2b1_regmass' : 'H2_b1_m',
    'ak4_h2b1_regpt' : 'H2_b1_ptRegressed',

    'ak4_h2b2_bRegRes' : 'H2_b2_bRegRes',
    'ak4_h2b2_bdisc' : 'H2_b2_deepJet',
    'ak4_h2b2_btag_L' : 'H2_b2_deepJet > 0.0490',
    'ak4_h2b2_btag_M' : 'H2_b2_deepJet > 0.2783',
    'ak4_h2b2_btag_T' : 'H2_b2_deepJet > 0.7100',
    'ak4_h2b2_eta' : 'H2_b2_eta',
    'ak4_h2b2_hflav' : 'H2_b2_hadronFlavour',
    'ak4_h2b2_mass' : 'H2_b2_m',
    'ak4_h2b2_phi' : 'H2_b2_phi',
    'ak4_h2b2_pt' : 'H2_b2_pt',
    'ak4_h2b2_regmass' : 'H2_b2_m',
    'ak4_h2b2_regpt' : 'H2_b2_ptRegressed',

    'dHH_H1_H2_deltaEta':'h1h2_deltaEta',
    'dHH_H1_H2_deltaPhi':'h1h2_deltaPhi',
    'dHH_H1_H2_deltaR':'h1h2_deltaR',
    'dHH_H1_pt':'H1unregressed_pt',   
    'dHH_H1_mass':'H1unregressed_m',
    'dHH_H1_regpt':'H1_pt',
    'dHH_H1_regmass':'H1_m',
    'dHH_H1b1_H1b2_deltaEta':'H1_bb_deltaEta',
    'dHH_H1b1_H1b2_deltaPhi':'H1_bb_deltaPhi',
    'dHH_H1b1_H1b2_deltaR':'H1_bb_deltaR',

    'dHH_H2_pt':'H2unregressed_pt',
    'dHH_H2_mass':'H2unregressed_m',
    'dHH_H2_regpt':'H2_pt',
    'dHH_H2_regmass':'H2_m',
    'dHH_H2b1_H2b2_deltaEta':'H2_bb_deltaEta',
    'dHH_H2b1_H2b2_deltaPhi':'H2_bb_deltaPhi',
    'dHH_H2b1_H2b2_deltaR':'H2_bb_deltaR',

    'dHH_HH_mass':'HHunregressed_m',
    'dHH_HH_pt':'HHunregressed_pt',
    'dHH_HH_regmass':'HH_m',
    'dHH_HH_regpt':'HH_pt',

    'dHH_NbtagT' : 'nBtagTightonMediumWP',
    'dHH_SumRegPtb' : 'sum_4b_pt',
    'dHH_SumRegResb' : 'sum_3b_bres',
    'dHH_absCosTheta_H1_inHHcm' : 'abs_costh_H1_ggfcm',
    'dHH_absCosTheta_H1b1_inH1cm' : 'abs_costh_H1_b1_h1cm',
    'dHH_maxdEtabb' : 'max_4b_deltaEta',
    'dHH_mindRbb' : 'min_4b_deltaR',
}

def map_to_nano(tree):
    tree.extend(**{
        nano_key : tree[bbbb_key]
        for nano_key, bbbb_key in bbbb_variable_map.items()
    })