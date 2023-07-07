from typing import Any


class AttrDict(dict):
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.update(*args, **kwargs)
    def __getattr__(self, key):
        return self[key]
    def __setattr__(self, key, value):
        self[key] = value

def _unzip_kwargs(**kwargs):
    if not any(kwargs): return []
    keys = list(kwargs.keys())
    values = list(kwargs.values())
    nobjs = len(values[0])
    return [ AttrDict({ key:value[i] for key,value in zip(keys,values) }) for i in range(nobjs) ]
    
def _zip_args(*args):
    if not any(args): return {}
    keys = list(args[0].keys())
    return AttrDict({ key:[ arg[key] for arg in args ] for key in keys })

class AttrArray:
    @staticmethod
    def init_attr(attr, init, size):
        if attr is None:
            return [init]*size
        if not isinstance(attr, list):
            attr = [attr]
        return (attr + size*[init])[:size]

    def __init__(self,**kwargs):
        nobjs = 0
        if any(kwargs):
            nobjs = len( list(kwargs.values())[0] )
            for key,value in kwargs.items(): 
                if not isinstance(value,(list)): value = self.init_attr(None,value,nobjs)
                default = False if isinstance(value[0],bool) else None
                kwargs[key] = self.init_attr(value,default,nobjs)
                
        kwargs["__id__"] = kwargs.get("__id__", list(range(nobjs)))
        
        self.kwargs = AttrDict(kwargs)
        self.args = _unzip_kwargs(**kwargs)
        
    @property
    def fields(self): return list(self.kwargs.keys())
        
    def unzip(self, fields=None):
        if fields is None: fields = self.fields 
        return self[fields]
    
    def filter(self, attr_filter):
        if isinstance(attr_filter,str):
            newargs = [ arg for arg in self if arg[attr_filter] ]
        elif callable(attr_filter):
            newargs = [ arg for arg in self if attr_filter(arg) ]
        return AttrArray(**_zip_args(*newargs))
    
    def split(self, attr_filter):
        split_t = self.filter(attr_filter)
        ids = split_t.__id__ if any(split_t) else []
        split_f = self.filter(lambda attr : not attr['__id__'] in ids )
        return split_t,split_f
        
    def __getattr__(self,key): 
        return self[key]
    
    def __getitem__(self,key):
        if isinstance(key,str): return self.kwargs[key]
        if isinstance(key,list) and isinstance(key[0],str):
            return { k:self[k] for k in key }
        if isinstance(key,list) and isinstance(key[0],int):
            return AttrArray(**_zip_args(*[ self[k] for k in key ]))
        return self.args[key]
    
    def __repr__(self): return f'<AttrArray {repr(self.args)} >'
    def __str__(self): return str(self.args)
    def __iter__(self): return iter(self.args)
    def __len__(self): return len(self.args)