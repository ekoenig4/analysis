import numpy as np
import awkward as ak
import multiprocessing as mp
import threading as th
from threading import Thread
from multiprocess.pool import ThreadPool
from functools import partial
import time

from tqdm import tqdm
# from ..rich_tools import tqdm

class ParallelMethod:
    def __init__(self):
        self.__time__ = time.time()
        self.__start_timing__ = []
        self.__run_timing__ = []
        self.__end_timing__ = []
    
    def start(self, *args, **kwargs):
        return dict()
    def run(self, *args, **kwargs):
        return dict()
    def end(self, *args, **kwargs):
        return 

    @property
    def start_timing(self):
        return ak.Array(self.__start_timing__)
    
    @property
    def run_timing(self):
        return ak.Array(self.__run_timing__)
    
    @property
    def end_timing(self):
        return ak.Array(self.__end_timing__)

    def __call__(self, *args, **kwargs):
        inputs, start_timing = self.__start__(0, args, kwargs)
        output, run_timing = self.__run__(inputs, start_timing)
        finished, end_timing = self.__end__(args, output, run_timing)

        self.__start_timing__.append(start_timing)
        self.__run_timing__.append(run_timing)
        self.__end_timing__.append(end_timing)

        return finished
    
    def parallel(self, iargs, pool=None, **kwargs):
        id, args = iargs[0], iargs[1:]
        inputs, start_timing = self.__start__(id, args, kwargs)
        result = pool.starmap(self.__run__, [(inputs, start_timing)])
        output, run_timing = list(result)[0]
        finished, end_timing = self.__end__(args, output, run_timing)

        self.__start_timing__.append(start_timing)
        self.__run_timing__.append(run_timing)
        self.__end_timing__.append(end_timing)

        return id, finished
    
    def __start__(self, id, args, kwargs):
        start = time.time() - self.__time__
        result = self.start(*args, **kwargs)
        worker = mp.current_process().name
        thread = th.current_thread().name
        end = time.time() - self.__time__

        timing = dict(
            id=id,
            start=start,
            end=end,
            worker=worker,
            thread=thread,
        )

        return result, timing

    def __run__(self, inputs, timing):
        start = time.time() - self.__time__
        result = self.run(**inputs)
        worker = mp.current_process().name
        thread = th.current_thread().name
        end = time.time() - self.__time__

        timing = dict(
            timing,
            start=start,
            end=end,
            worker=worker,
            thread=thread,
        )

        return result, timing
    
    def __end__(self, args, outputs, timing):
        start = time.time() - self.__time__

        if outputs:
            result = self.end(*args, **outputs)
        else:
            result = self.end(*args)
            
        worker = mp.current_process().name
        thread = th.current_thread().name
        end = time.time() - self.__time__
        
        timing = dict(
            timing,
            start=start,
            end=end,
            worker=worker,
            thread=thread,
        )

        return result, timing

class ObjThread(Thread):
    def __init__(self, obj, obj_function):
        super().__init__()
        self.obj = obj
        self.obj_function = obj_function
    def run(self):
        self.result = self.obj_function(self.obj)

class ThreadManager:
    def __init__(self, objs, obj_function):
        self.threads = [ ObjThread(obj, obj_function) for obj in objs ]

    def __enter__(self):
        return self
    
    def run(self, report=False):
        threads = self.threads
        for thread in threads: thread.start()

        waiting_for = list(range(len(threads)))
        pbar = tqdm(total=len(waiting_for)) if report else None
        while len(waiting_for) > 0:
            for i in waiting_for:
                thread = threads[i]
                if thread.is_alive(): continue

                waiting_for.remove(i)
                if pbar: pbar.update(1)
                thread.join()
        if pbar: pbar.close()

        # for thread in threads: thread.join()
        return [thread.result for thread in self.threads]
        
    def __exit__(self, *args):
        return

class ObjTransform:
    def __init__(self,**kwargs):
        self.__dict__.update(**kwargs)

        if hasattr(self,'init') and callable(self.init):
            self.init()
        
    def __getattr__(self, key):
        if key in self.__dict__: return self.__dict__[key]
        return None
        
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
    
def get_function_name(obj):
    if hasattr(obj,'__name__'): return obj.__name__
    if hasattr(obj,'__class__'): return obj.__class__.__name__
    return str(obj)

class ObjIter:
    @classmethod
    def interweave(cls, *objiters):
        return cls([ obj for objs in zip(*objiters) for obj in objs ])


    def __init__(self,objs):
        self.objs = list(objs)

    def __getstate__(self):
        return self.objs
    def __setstate__(self, state):
        self.objs = state

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

    def __format__(self, spec):
        return '['+', '.join([ format(value, spec) for value in self.objs ]) + ']'
    
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
    @property
    def flat(self):
        return ObjIter([ obj for objs in self.objs for obj in objs ])

    def zip(self, other):
        return ObjIter(list(zip(self.objs, other.objs)))
    
    def filter(self,obj_filter):
        return ObjIter(list(filter(obj_filter,self)))
    
    def split(self,obj_filter):
        split_t = self.filter(obj_filter)
        split_f = self.filter(lambda obj : obj not in split_t)
        return split_t,split_f

    def parallel_apply(self, obj_function, report=False, pool=None):
        # if not isinstance(obj_function, ParallelMethod):
        #     print("Warning: parallel_apply is not recommended for non-ParallelMethod functions")
        #     return self.pool_apply(obj_function, report=report, pool=pool)
        
        # thread_pool = ThreadPool(len(self))
        thread_pool = ThreadPool(2*pool._processes if pool else len(self))

        parallel_function = partial(obj_function.parallel, pool=pool)
        result = thread_pool.imap_unordered( parallel_function, enumerate(self.objs), chunksize=1 )
        
        if report:
            result = tqdm(result, total=len(self), desc=get_function_name(obj_function))

        result = map(lambda x : x[1], sorted(result, key=lambda x: x[0]))
        return ObjIter(result)
    
    def pool_apply(self, obj_function, report=False, pool=None):
        if isinstance(obj_function, ParallelMethod):
            return self.parallel_apply(obj_function, report=report, pool=pool)

        if pool is None:
            pool = ThreadPool(len(self))

        result = pool.imap(obj_function, self.objs, chunksize=1)

        if report:
            result = tqdm(result, total=len(self), desc=get_function_name(obj_function))

        result = list( result )

        return ObjIter(result)

    
    def apply(self, obj_function, report=False, parallel=None, **kwargs):

        if parallel:
            return self.parallel_apply(obj_function, report=report, **kwargs)

        it = tqdm(self, desc=get_function_name(obj_function)) if report else self
        out = ObjIter([ obj_function(obj) for obj in it ])
        
        return out
        
    def copy(self):
        return ObjIter([obj.copy() for obj in self])