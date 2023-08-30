from ..ak_tools import build_p4, build_collection
from ..hepUtils import calc_dr_p4
import awkward as ak
import numpy as np

def match_ak4_gen(tree):
    build_collection(tree, 'genj_H\db\d', 'genjet_b')
    build_collection(tree, 'ak4_h\db\d', 'ak4jet_b')
    tree.extend(
        ak4jet_b_m=tree.ak4jet_b_mass,
    )

    genjet = build_p4(tree, 'genjet_b')
    ak4jet = build_p4(tree, 'ak4jet_b')
    ak4_gen_dr = calc_dr_p4(ak4jet, genjet[:,None])

    flat_ak4_gen_dr = ak.flatten(ak4_gen_dr, axis=2)
    flat_ak4_gen_dr_index = ak.local_index(flat_ak4_gen_dr, axis=1)
    flat_ak4_index = flat_ak4_gen_dr_index // 4
    flat_gen_index = flat_ak4_gen_dr_index % 4

    best_matches = []
    best_matches_dr = []

    flat_match_mask = ak.zeros_like(flat_ak4_gen_dr, dtype=bool)

    for i in range(4):
        next_flat_ak4_gen_dr = ak.where(flat_match_mask, np.inf, flat_ak4_gen_dr)
        next_index = ak.argmin( next_flat_ak4_gen_dr, axis=1)
        next_min_dr = ak.min( next_flat_ak4_gen_dr, axis=1)

        next_ak4_index = next_index // 4
        next_gen_index = next_index %  4

        best_matches.append( next_index )
        best_matches_dr.append( next_min_dr )

        is_next_ak4 = flat_ak4_index == next_ak4_index
        is_next_gen = flat_gen_index == next_gen_index
        is_next_ak4_gen = is_next_ak4 | is_next_gen

        flat_match_mask = flat_match_mask | is_next_ak4_gen

    best_matches = np.stack(best_matches, axis=1)
    best_matches_dr = np.stack(best_matches_dr, axis=1)
    best_matches_dr4_mask = best_matches_dr < 0.4
    ak4jet_ak4id = best_matches // 4

    ak4jet_genid = ak.where(best_matches_dr4_mask, best_matches % 4, -1)

    ak4jet_b_index = ak.argsort(ak4jet_ak4id, axis=1)
    ak4jet_b_genid = ak4jet_genid[ak4jet_b_index]
    ak4jet_b_gendr = best_matches_dr[ak4jet_b_index]

    ak4j1_b_hid = ak4jet_b_genid[:,::2] // 2
    ak4j2_b_hid = ak4jet_b_genid[:,1::2] // 2

    h_genid = ak.where( ak4j1_b_hid == ak4j2_b_hid, ak4j1_b_hid, -1)

    tree.extend(
        ak4jet_b_genid=ak4jet_b_genid,
        ak4jet_b_gendr=ak4jet_b_gendr,

        dHH_H1_genid=h_genid[:,0],
        dHH_H2_genid=h_genid[:,1],
        
        nfound_select=ak.sum(ak4jet_b_genid >= 0, axis=1),
        nfound_paired=ak.sum(h_genid >= 0, axis=1),
    )