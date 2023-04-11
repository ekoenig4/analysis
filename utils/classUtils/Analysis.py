import inspect
from collections import defaultdict
from .ObjIter import ObjIter
from .Stopwatch import Stopwatch
from tqdm import tqdm
from termcolor import colored
import time

import re
def find_all_undeclared(object):
    source = inspect.getsource(object)
    members = set(vars(object).keys())
    ordered_attrs = re.findall('self\.(\w+)',source)
    attrs = set(ordered_attrs)
    declared = set(re.findall('self\.(\w+)\s*=', source))
    undeclared = set.difference(attrs, declared).difference(members)
    return sorted(undeclared, key=ordered_attrs.index)

def f7(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]
class MethodDependency:
    _required_graphs = defaultdict(dict)
    _dependency_graphs = defaultdict(lambda:defaultdict(list))

    @classmethod
    def dependency(cls, *required):
        required = [ f.__name__ for f in required ]

        def set_dependency(method, required=required):
            cls._dependency_graphs[method.__qualname__.split('.')[0]][method.__name__] = required
            return method
        return set_dependency

    @classmethod
    def required(cls, method):
        cls._required_graphs[method.__qualname__.split('.')[0]][method.__name__] = None
        return method

    @classmethod
    def get_dependency(cls, analysis):
        if not isinstance(analysis, str):
            analysis = analysis.__name__
        return cls._dependency_graphs[analysis]


    @classmethod
    def get_required(cls, analysis):
        if not isinstance(analysis, str):
            analysis = analysis.__name__
        return list(cls._required_graphs[analysis].keys())

    def __init__(self, analysis, methods):
        required_graph = MethodDependency.get_required(analysis.__class__)
        dependency_graph = MethodDependency.get_dependency(analysis.__class__)

        for required in required_graph:
            idx = methods.index(required)
            for method in methods[idx+1:]:
                dependency_graph[method] = f7(dependency_graph[method] + [required])

        for method, dependency in dependency_graph.items():
            dependency_graph[method] = sorted(dependency, key=methods.index)
        self.dependency_graph = dependency_graph

    def build_runlist(self, runlist):
        graph = self.dependency_graph
        def get_dependence(method):
            method_dependency = graph.get(method, dict())
            runlist = []
            for dependence in method_dependency:
                runlist += get_dependence(dependence)
            runlist += [method]
            return f7(runlist)
            
        full_runlist = []
        for method in runlist:
            full_runlist = f7(full_runlist+get_dependence(method))
        return full_runlist


required = MethodDependency.required
dependency = MethodDependency.dependency

class AnalysisMethod:
    def __init__(self, analysis, method, disable=False):
        self.analysis = analysis 
        self.method = method
        self.enabled = True
        self.disable = disable
        self._build_args()
        
    def _build_args(self):
        params = inspect.signature(self.method).parameters
        self.args = []
        self.use_kwargs = False
        for key, param in params.items():
            if key == 'self': continue
            if param.kind is inspect.Parameter.VAR_KEYWORD: 
                self.use_kwargs = True 
                break
            self.args.append(key)

    def __call__(self):
        if self.disable or not self.enabled: return
        if self.use_kwargs:
            result = self.method(self.analysis, **vars(self.analysis))
        else:
            result = self.method(self.analysis, *[ getattr(self.analysis, key, None) for key in self.args ])
        self.enabled = False
        return result
    @property
    def __name__(self): return self.method.__name__
    def __str__(self): 
        status =     colored('done'.center(8),'green')
        if self.disable:
            status = colored('disabled'.center(8), 'red', attrs=['dark'])
        elif self.enabled:
            status = colored('pending'.center(8), 'blue')

        name = colored(self.__name__, 'white', attrs=['dark'] if self.disable else [])
        args = colored(', '.join(self.args), 'white', attrs=['dark'])
        return f"<[{status}] {name}({args})>"
    def __repr__(self): return str(self)
    def run(self):
        self.enabled = True
        return self()

class MethodList:
    def __init__(self, analysis, runlist=None, **methods):
        self.analysis = analysis

        def _check_runlist_(method):
            if runlist is None: return True
            return method in runlist

        self.methods = { key: AnalysisMethod(analysis, method, disable=not _check_runlist_(key)) for key,method in methods.items() }
    @property
    def keys(self): return list(self.methods.keys())
    @property
    def values(self): return list(self.methods.values())
    def __str__(self):
        lines = [ f"{i:>5}: {str(method)}" for i, method in enumerate(self.methods.values()) ]
        return '<MethodList\n'+'\n'.join(lines)+'\n>'
    def __repr__(self): return str(self)
    def __getiter__(self):
        return iter(self.methods.values())
    def items(self): return self.methods.items()
    def __getitem__(self, key):
        if isinstance(key, str):
            return self.methods[key]
        elif isinstance(key, int):
            return self.values[key]
        elif isinstance(key, list):
            methods = [ self[k] for k in key ]
            methods = { method.__name__:method for method in methods }
        elif isinstance(key, slice):
            methods = { key: self.methods[key] for key in self.keys[key] }

        method_list = MethodList(self.analysis)
        method_list.methods = methods
        return method_list


class Analysis:
    def __init__(self, name=None, runlist=None, ignore_error=True, signal=ObjIter([]), bkg=ObjIter([]), data=ObjIter([]), **kwargs):
        self.name = name
        if name is None:
            self.name = str(self.__class__.__name__)
            
        self.signal = signal 
        self.bkg = bkg 
        self.data = data

        # self.trees = signal + bkg + data

        methods = { key : method for key, method in vars(self.__class__).items() if (not key.startswith('_') and callable(method)) }
        self.dependency = MethodDependency(self, list(methods.keys()))

        if runlist is not None: runlist = self.build_runlist(runlist)
        self.methods = MethodList(self, runlist=runlist, **methods)
        self.ignore_error = ignore_error

        self.__dict__.update(**kwargs)

    @property
    def trees(self): return self.signal + self.bkg + self.data

    def run(self, runlist=None, **kwargs):
        if runlist is None: 
            runlist = [ key for key,method in self.methods.items() if not method.disable ]
        runlist = self.build_runlist(runlist)

        stopwatch = Stopwatch()
        for i, (key, method) in enumerate(self.methods.items()): 
            if key not in runlist: 
                print(f'{stopwatch} [{colored("skipping","white", attrs=["dark"])}] {colored(key, "white", attrs=["dark"])}')
                continue
            print(f'{stopwatch} [{colored("running","green")}] {key}')

            if self.ignore_error:
                try:
                    method()
                except Exception as e:
                    print(f'{stopwatch} [{colored("error","red")}] {e}\n')
            else:
                method()
        print(f'{stopwatch} [{colored("finished","green")}]')

    def __getattr__(self, key): return self.__dict__.get(key, None)

    @classmethod
    def get_args(cls):
        args = find_all_undeclared(cls)
        return args

    def build_runlist(self, runlist):
        return self.dependency.build_runlist(runlist)

    def __repr__(self):
        lines = [
            f'Analysis: {self.name}',
        ] + [
            repr(self.methods)
        ] 
        return '\n'.join(lines)
