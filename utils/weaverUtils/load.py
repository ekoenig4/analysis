import numpy as np
import awkward as ak
import glob, os
import awkward0 as ak0

def load_from_ak0(toload, fields=['scores']):
    fields = {
    field:np.concatenate([ np.array(ak0.load(fn)[field], dtype=float) for fn in toload ])
    for field in fields
    }

    return fields

def load_from_root(toload, fields=['scores']):
    import uproot

    def load_fields(fn, fields):
        with uproot.open(fn) as f:
            ttree = f['Events']
            return { field:ttree[field].array() for field in fields }

    arrays = [
        load_fields(fn, fields)
        for fn in toload
    ]

    return {
        field:np.concatenate([ array[field] for array in arrays ])
        for field in fields
    }

def load_output(tree, model=None, fields=['scores']):
    predict_path = os.path.join(model,"predict_output")
    if not os.path.exists(predict_path):
        raise ValueError(f'No predict_output found for model {model}.')

    rgxs = [ os.path.basename(os.path.dirname(fn.fname))+"_"+os.path.basename(fn.fname) for fn in tree.filelist ]
    toload = [ fn for rgx in rgxs for fn in glob.glob( os.path.join(predict_path,rgx) ) ]
    if any(toload): return load_from_root(toload, fields=fields)

    print(f'No root output found for {rgxs[0]}... trying without base...')

    rgxs = [ os.path.basename(os.path.dirname(fn.fname))+'.root' for fn in tree.filelist ]
    toload = [ fn for rgx in rgxs for fn in glob.glob( os.path.join(model,"predict_output",rgx) ) ]
    if any(toload): return load_from_root(toload, fields=fields)

    print(f'No root output found for {rgxs[0]}... trying awkd...')

    rgxs = [ os.path.basename(os.path.dirname(fn.fname))+"_"+os.path.basename(fn.fname)+".awkd" for fn in tree.filelist ]
    toload = [ fn for rgx in rgxs for fn in glob.glob( os.path.join(model,"predict_output",rgx) ) ]

    if any(toload): return load_from_ak0(toload, fields=fields)

    raise ValueError(f'No weaver output found for model {model}.')