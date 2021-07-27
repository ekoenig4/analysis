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
