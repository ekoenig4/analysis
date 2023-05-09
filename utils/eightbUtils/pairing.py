import vector
import awkward0 as ak0
import awkward as ak
import glob, os

from ..ak_tools import *
from ..hepUtils import build_all_dijets
from ..combinatorics import combinations

def reconstruct(jet_p4, assignment, tag=''):
    if tag and not tag.endswith('_'): tag += '_'

    j_p4 = jet_p4[assignment]

    h_j1_arg, h_j2_arg = assignment[:, ::2], assignment[:, 1::2]
    j1_p4, j2_p4 = jet_p4[h_j1_arg], jet_p4[h_j2_arg]

    h_p4 = j1_p4 + j2_p4
    h_signalId = ak.where( j1_p4.signalId//2 == j2_p4.signalId//2, j1_p4.signalId//2, -1 )

    h1_p4, h2_p4 = h_p4[:, ::2], h_p4[:, 1::2]
    h1_signalId, h2_signalId = h_signalId[:, ::2], h_signalId[:, 1::2]

    y_p4 = h1_p4 + h2_p4
    y_signalId = ak.where( h1_signalId//2 == h2_signalId//2, h1_signalId//2, -1 )

    x_p4 = y_p4[:,0] + y_p4[:,1]
    x_signalId = ak.where( y_signalId[:,0]//2 == y_signalId[:,1]//2, y_signalId[:,0]//2, -1 )

    y_pt_order = ak_rank(y_p4.pt, axis=1)
    h_pt_order = ak_rank(h_p4.pt, axis=1)
    j_pt_order = ak_rank(jet_p4.pt, axis=1)

    y_h_pt_order = h_pt_order + 10*y_pt_order[:,[0,0,1,1]]
    y_h_j_pt_order = j_pt_order + 100*y_h_pt_order[:,[0,0,1,1,2,2,3,3]]

    j_order = ak.argsort(y_h_j_pt_order, axis=1, ascending=False)
    h_order = ak.argsort(y_h_pt_order, axis=1, ascending=False)
    y_order = ak.argsort(y_pt_order, axis=1, ascending=False)
    
    j_p4 = j_p4[j_order]
    h_p4 = h_p4[h_order]
    y_p4 = y_p4[y_order]
    h_signalId = h_signalId[h_order]
    y_signalId = y_signalId[y_order]

    p4vars = ['pt','eta','phi','m']
    return dict(
        **{f'{tag}x_{var}': getattr(x_p4, var) for var in p4vars},
        **{f'{tag}x_signalId': x_signalId},
        **{f'{tag}y_{var}': getattr(y_p4, var) for var in p4vars},
        **{f'{tag}y_signalId': y_signalId},
        **{f'{tag}h_{var}': getattr(h_p4, var) for var in p4vars},
        **{f'{tag}h_signalId': h_signalId},
        **{f'{tag}j_{var}': getattr(j_p4, var) for var in j_p4.fields},
    )

def load_weaver_output(tree, model=None, fields=['scores']):
  rgxs = [ os.path.basename(os.path.dirname(fn.fname))+"_"+os.path.basename(fn.fname)+".awkd" for fn in tree.filelist ]
  toload = [ fn for rgx in rgxs for fn in glob.glob( os.path.join(model,"predict_output",rgx) ) ]

  fields = {
    field:np.concatenate([ np.array(ak0.load(fn)[field], dtype=float) for fn in toload ])
    for field in fields
  }
  return fields

def load_feynnet_assignment(tree, model, extra=[], reco_event=True):
    fields = ['maxcomb','maxscore','minscore'] + extra
    ranker = load_weaver_output(tree, model, fields=fields)
    
    score, assignment, minscore = ranker['maxscore'], ranker['maxcomb'], ranker['minscore']
    tree.extend(feynnet_maxscore=score, feynnet_minscore=minscore, **{f'feynnet_{field}':ak.from_regular(ranker[field]) for field in extra})

    if not reco_event: return

    assignment = ak.from_regular(assignment.astype(int))

    jet_p4 = build_p4(tree, prefix='jet', use_regressed=True, extra=['signalId', 'btag'])
    reconstruction = reconstruct(jet_p4, assignment)
    tree.extend(**reconstruction)

def load_true_assignment(tree):
    jet_p4 = build_p4(tree, prefix='jet', use_regressed=True, extra=['signalId', 'btag'])

    true_assignment = ak.argsort(jet_p4.signalId, axis=1)
    true_reconstruction = reconstruct(jet_p4, true_assignment, tag='true_')
    tree.extend(**true_reconstruction)