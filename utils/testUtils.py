from . import *


print_nice = lambda array : [ (int(elem) if type(elem) is bool else elem) for elem in array ]

def is_iter(array):
    try:
        it = iter(array)
    except TypeError: return False
    return True

def check(selections,fields,ie=5):
    for field in fields:
        print(f"--- {field} ---")
        for selection in selections:
            if hasattr(selection,field): 
                value = getattr(selection,field)
                if is_iter(value): value = value[ie]
                tag = selection.tag
            else: 
                value = selection[field]
                if is_iter(value): value = value[ie]
                tag = "event"
            printout = str(value)
            if is_iter(value): printout=print_nice(value)
            print(f"{tag:<15}: {printout}")
            
def icheck(arrays,ie=None,mask=None,imax=9):
    for i,array in enumerate(arrays):
        if i > imax: return
        if mask is not None: array = array[mask]
        if ie is not None: array = array[ie]
            
        if is_iter(array) and len(array) < 10: array = print_nice(array)
        print(array)
        
def print_bovers(selections):
    signal = selections[0]
    bkgs = selections[1:]
    scale =sum( [ ak.sum(bkg["scale"]) for bkg in bkgs ] )/ak.sum(signal["scale"])
    print(f"Bkg/Signal: {scale:0.2f}")
    for selection in selections:print(selection)
    print("------")
    
def print_raw_info(tree,lumikey=2018):
    lumi,tex = lumiMap[lumikey]
    is_data = tree.is_data
    for sample,xsec,total,raw in zip(tree.samples,tree.xsecs,tree.total_events,tree.raw_events):
        sample = sample.replace("_","\_")
        expect = lumi*xsec*raw/total if not is_data else raw
        print(f"{sample} & {xsec} & {total:.0f} & {raw} & {expect:0.2f}\\\\")
