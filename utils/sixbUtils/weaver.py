from ..ak_tools import get_collection, build_p4
from ..hepUtils import build_all_dijets, calc_dr_p4
import numpy as np
import awkward as ak
import os, glob

def load_sixb_weaver(tree, model, fields=['scores']):
    import awkward0 as ak0

    rgxs = [ os.path.basename(os.path.dirname(fn.fname))+".root.awkd" for fn in tree.filelist ]

    toload = [ fn for rgx in rgxs for fn in glob.glob( os.path.join(model,"predict_output",rgx) ) ]

    if not any(toload):
        rgxs = [ os.path.basename(os.path.dirname(fn.fname))+'_'+os.path.basename(fn.fname)+".awkd" for fn in tree.filelist ]
        toload = [ fn for rgx in rgxs for fn in glob.glob( os.path.join(model,"predict_output",rgx) ) ]


    fields = {
        field:np.concatenate([ np.array(ak0.load(fn)[field], dtype=float) for fn in toload ])
        for field in fields
    }
    return fields
    # model = f'{model}/predict_output/{t.sample}.awkd'

    # with ak0.load(model) as f_ak:
    #     fields = {
    #         field:np.concatenate([ np.array(f_ak[field], dtype=float) ])
    #         for field in fields
    #     }
    # return fields

def reco_yh_trih(tree, index):
    index = np.stack([index[:,::2], index[:, 1::2]], axis=2)
    jet_index = ak.from_regular(index.astype(int))
    jets = get_collection(tree, 'jet', named=False)
    
    build_all_dijets(tree, pairs=jet_index, name='higgs', ordered='pt')
    higgs = get_collection(tree, 'higgs', named=False)
    hp4 = build_p4(higgs, extra=['signalId'])

    h_idx = ak.argsort((higgs.localId+1)//2,axis=-1)

    hx_idx = h_idx[:,:1]
    h1_idx = h_idx[:,1:2]
    h2_idx = h_idx[:,2:]
    hx = hp4[hx_idx][:,0]
    h1 = hp4[h1_idx][:,0]
    h2 = hp4[h2_idx][:,0]

    y_p4 = h1 + h2
    higgs_dr = calc_dr_p4(h1, h2)
    y = dict(
        m=y_p4.m,
        pt=y_p4.pt,
        eta=y_p4.eta,
        phi=y_p4.phi,
        higgs_dr=higgs_dr,
        h1Idx=h1_idx,
        h2Idx=h2_idx,
    )
    tree.extend(
        **{
            f'HX_{field}': hx[field]
            for field in ('pt','m','eta','phi','signalId')

        },
        **{
            f'H1_{field}': h1[field]
            for field in ('pt','m','eta','phi','signalId')

        },
        **{
            f'H2_{field}': h2[field]
            for field in ('pt','m','eta','phi','signalId')

        },
        **{
            f'Y_{field}':array
            for field, array in y.items()
        }
    )

def load_yh_trih_ranker(tree, model):
    ranker = load_sixb_weaver(tree, model, fields=['maxcomb','maxlabel','maxscore','minscore','scores'])
    score, index, label = ranker['maxscore'], ranker['maxcomb'], ranker['maxlabel']
    minscore = ranker['minscore']

    scores = ranker['scores'].reshape(-1, 45)
    tree.extend(yh_trih_score=score, yy_trih_label=label, yh_trih_scores=ak.from_regular(scores), yh_trih_minscore=minscore)

    reco_yh_trih(tree, index)    