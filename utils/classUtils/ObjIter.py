import numpy as np
import awkward as ak
from tqdm import tqdm
from typing import Callable

def _chunks_(total_events, batches):
    k, m = divmod(total_events, batches)
    for i in range(batches):
        yield i*k+min(i, m), (i+1)*k+min(i+1, m)

def chunk_method(array: ak.Array, array_method: Callable, batches=25, events=None, report=False) -> ak.Array:
    """Apply a method on an array in chunks. The final result is then concatenated together.

    Args:
        array (ak.Array): Awkward Array like structure
        array_method (Callable): Method that takes an array as the first argument and returns an array
        batches (int, optional): Number of batches to chunk the array into. Defaults to None.
        events (int, optional): Approximate number of events per chunk. Defaults to None.
        report (bool, optional): Gives TQDM reporting. Defaults to True.

    Returns:
        ak.Array: Awkward Array that is a concatenation of the results of the array_method
    """
    total_events = len(array)
    if batches:
        batches = batches
    if events:
        batches = total_events//events

    it = enumerate(_chunks_(total_events, batches))
    if report:
        it = tqdm(it, total=batches)

    builder = [array_method(array[start:stop]) for i, (start, stop) in it]
    return ak.concatenate(builder)


class ObjTransform:
    def __init__(self,**kwargs):
        self.__dict__.update(**kwargs)

        if hasattr(self,'init') and callable(self.init):
            self.init()
class MethodIter:
    def __init__(self, objs, calls):
        self.objs = objs
        self.calls = calls
        self.calliter = zip(objs, calls)

    def __str__(self): return str(self.calls)
    def __iter__(self): return iter(self.calls)
    def __getitem__(self, key): return self.calls[key]

    def __call__(self, *a, args=lambda t: [], kwargs=lambda t: {}, **kw):
        f_args, f_kwargs = args, kwargs
        if not callable(f_args):
            def f_args(t): return args
        if not callable(f_kwargs):
            def f_kwargs(t): return kwargs

        def build_args(t): return list(a)+list(f_args(t))
        def build_kwargs(t): return dict(**f_kwargs(t), **kw)
        out = [call(*build_args(t), **build_kwargs(t)) for t, call in self.calliter]
        return ObjIter(out)

    def zip(self, objiter, *a, args=lambda t:[], kwargs=lambda t:{}, **kw):
        f_args, f_kwargs = args, kwargs
        if not callable(f_args):
            def f_args(t): return args
        if not callable(f_kwargs):
            def f_kwargs(t): return kwargs

        def build_args(t): return list(a)+list(f_args(t))
        def build_kwargs(t): return dict(**f_kwargs(t), **kw)
        out = [call(obj, *build_args(t), **build_kwargs(t)) for (t, call), obj in zip(self.calliter, objiter) ]
        return ObjIter(out)

    
def get_slice(obj,slices):
    if len(slices) == 1:
        return obj[slices[0]]
    if len(slices) == 2:
        return obj[slices[0],slices[1]]
    if len(slices) == 3:
        return obj[slices[0],slices[1],slices[2]]

class ObjIter:
    def __init__(self,objs):
        self.objs = list(objs)

    def __len__(self): return len(self.objs)
    def __str__(self): return str(self.objs)
    def __iter__(self): return iter(self.objs)
    def __repr__(self): return repr(self.objs)
    def __getitem__(self, key): 
        if isinstance(key,list):
            return ObjIter([ self.objs[k] for k in key ])
        if isinstance(key,tuple):
            objs = self.objs[key[0]] if isinstance(key[0],slice) else [ self.objs[k] for k in key[0] ]
            return ObjIter([ get_slice(obj,key[1:]) for obj in objs ])
        if isinstance(key,slice): return ObjIter(self.objs[key])
        if isinstance(key,int): return self.objs[key]
        if isinstance(key,str): return getattr(self, key)
        return ObjIter(self.objs[key])

    def __getattr__(self, key):
        attriter = [getattr(obj, key) for obj in self]
        if callable(attriter[0]):
            attriter = MethodIter(self.objs, attriter)
        else:
            attriter = ObjIter(attriter)
        return attriter
        
    def __add__(self,other):
        if type(other) == list: other = ObjIter(other)
        return ObjIter(self.objs+other.objs)
    
    @property
    def numpy(self): return np.array(self.objs) 
    @property
    def npy(self): return np.array(self.objs)
    @property
    def awkward(self): return ak.from_regular(self.objs)
    @property
    def awk(self): return ak.from_regular(self.objs)
    @property
    def list(self): return self.objs
    @property
    def cat(self): return ak.concatenate(self.objs)

    def zip(self, other):
        return ObjIter(list(zip(self.objs, other.objs)))
    
    def filter(self,obj_filter):
        return ObjIter(list(filter(obj_filter,self)))
    
    def split(self,obj_filter):
        split_t = self.filter(obj_filter)
        split_f = self.filter(lambda obj : obj not in split_t)
        return split_t,split_f
    
    def apply(self,obj_function, report=False):
        it = tqdm(self) if report else self

        out = ObjIter([ obj_function(obj) for obj in it ])
        return out
        
    def copy(self):
        return ObjIter([obj.copy() for obj in self])