from collections import defaultdict
import numpy as np
import awkward as ak
import glob, os

def load_from_ak0(toload, fields=['scores']):
    import awkward0 as ak0
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

def get_rgx(fn):
    return os.path.basename(os.path.dirname(fn.fname))

def rgx_with_base(fn, rgx):
    return rgx + "_" + os.path.basename(fn.fname)

def rgx_with_year(fn, rgx):
    years = {
        'Summer2016':'2016postVFP',
        'Summer2016/preVFP':'2016preVFP',
        'Summer2017':'2017',
        'Summer2018':'2018',
    }

    year = next( (key for year, key in years.items() if year in fn.fname), None )
    if year is None: return rgx
    return year + '/' + rgx

def load_output(tree, model=None, fields=['scores'], try_base=True, try_year=False):
    predict_path = os.path.join(model,"predict_output")
    if not os.path.exists(predict_path):
        raise ValueError(f'No predict_output found for model {model}.')
    
    try_rgxs = []
    if try_base: try_rgxs += [rgx_with_base]
    if try_year: try_rgxs += [rgx_with_year]

    rgxs = [ get_rgx(fn) for fn in tree.filelist ]
    for try_rgx in try_rgxs:
        rgxs = [ try_rgx(fn, rgx) for fn, rgx in zip(tree.filelist, rgxs) ]

    toload = [ fn for rgx in rgxs for fn in glob.glob( os.path.join(predict_path,rgx)+'*' ) ]
    if not any(toload): 
        raise ValueError(f'No weaver output found for model {model}. With rgxs {rgxs}')
        
    load = load_from_root if all( fn.endswith('.root') for fn in toload ) else load_from_ak0
    return load(toload, fields=fields)
