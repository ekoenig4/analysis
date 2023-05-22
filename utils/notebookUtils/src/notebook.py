from .cell_dependency import CellDependency
from .cell import Cell

# from ...rich_tools import print
# from rich import print

from argparse import ArgumentParser

import time
import datetime

class Stopwatch:
    def __init__(self):
        self.start = time.perf_counter()
        self.previous_timestamp = -1

    def __str__(self):
        now = time.perf_counter() - self.start 
        timestamp = str(datetime.timedelta(seconds=now))
        timestamp = timestamp.split('.')[0]

        if timestamp == self.previous_timestamp: return ' '*len(timestamp)
        
        self.previous_timestamp = timestamp
        return timestamp


def add_to_list(i, method, disable_list=[], only_list=[]):
    """ 
    Check if a method should be added to the list of methods to run
    """
    import re 

    for disable in disable_list:
        if disable.isdigit() and int(disable) == i: return False
        if re.match(f'^{disable}$', method): return False 

    for only in only_list:
        if only.isdigit() and int(only) == i: return True
        if re.match(f'^{only}$', method): return True 

    return False

def add_parser_from_loadlist(cls, parser):
    loadlist = cls.__load__ + [cls]
    for notebook in loadlist:
        group = parser.add_argument_group(f'{notebook.__name__} arguments')
        notebook.add_parser(group)

def add_parser_from_inheritance(cls, parser):
    # check if add_parser is overriden
    overides_add_parser = True

    for base in cls.__bases__:
        if base == Notebook: continue
        add_parser_from_inheritance(base, parser)
        # check if add_parser is overriden
        overides_add_parser &= cls.add_parser != base.add_parser
    
    if not overides_add_parser: return
    group = parser.add_argument_group(f'{cls.__name__} arguments')
    cls.add_parser(group)

def add_methods_from_loadlist(cls, methods={}):
    loadlist = cls.__load__ + [cls]
    for notebook in loadlist:
            methods.update({ key : method for key, method in vars(notebook).items() if (not key.startswith('_') and callable(method)) })
    return methods

def add_methods_from_inheritance(cls, methods={}):
    for base in cls.__bases__:
        if base == Notebook: continue
        add_methods_from_inheritance(base, methods)
    methods.update({ key : method for key, method in vars(cls).items() if (not key.startswith('_') and callable(method)) })
    return methods

def add_dependencies_from_loadlist(cls, method_list, dependencies=[]):
    loadlist = cls.__load__ + [cls]
    dependencies = [ CellDependency(notebook, method_list) for notebook in loadlist ]
    return CellDependency.merge(*dependencies)

def add_dependencies_from_inheritance(cls, method_list, dependencies=[]):
    for base in cls.__bases__:
        if base == Notebook: continue
        add_dependencies_from_inheritance(base, method_list, dependencies)

    dependencies.append(CellDependency(cls, method_list))
    return CellDependency.merge(*dependencies)

class Notebook:
    """
    A notebook of methods. Similar to a jupyter notebook.
    """
    __load__ = [] # depcrecated in favor of inheritance

    @staticmethod
    def init_parser(parser):
        parser.add_argument('--ignore-error', action='store_true', help='ignore errors from cells')
        parser.add_argument('--dry-run', action='store_true', help='dry run the notebook without executing cells')
        parser.add_argument('--only', nargs='+', help='only run these cells', default=[])
        parser.add_argument('--disable', nargs='+', help='disable these cells', default=[])
        return parser

    @staticmethod
    def add_parser(parser):
        return parser

    @classmethod
    def from_parser(cls, parser=None, **kwargs):
        parser = parser or ArgumentParser()

        cls.init_parser(parser)

        # add_parser_from_loadlist(cls, parser)
        add_parser_from_inheritance(cls, parser)

        args = parser.parse_args()
        return cls(**kwargs, **vars(args))

    def __init__(self, ignore_error=False, dry_run=False, only=[], disable=[], **kwargs):
        methods = {}

        # methods = add_methods_from_loadlist(self, methods)
        methods = add_methods_from_inheritance(self.__class__, methods)
        method_list = list(methods.keys())

        # self._dependency = add_dependencies_from_loadlist(self, method_list)
        self._dependency = add_dependencies_from_inheritance(self.__class__, method_list)
        
        self._cells = dict()
        self.add(**methods)

        self.build_runlist(method_list, only_list=only, disable_list=disable)

        self.ignore_error = ignore_error
        self.dry_run = dry_run
        self.__dict__.update(**kwargs)

    @property
    def __name__(self): return self.__class__.__name__

    def __getattr__(self, key): return self.__dict__.get(key, None)
    
    @property
    def namespace(self):
        return { key:value for key, value in self.__dict__.items() if not key.startswith('_') }
    
    def hello(self):
        self.print_namespace()
        print(str(self))
    
    def print_namespace(self):
        print("---Arguments---")
        for key, value in self.namespace.items():
            print(f"{key} = {value}")
        print()
    
    def add(self, **methods):
        for key, method in methods.items():
            self._cells[key] = Cell(self, method)

    def build_runlist(self, runlist, only_list=[], disable_list=[]):
        if only_list or disable_list or None:
            runlist = [ key for i, key in enumerate(runlist) if add_to_list(i, key, disable_list=disable_list, only_list=only_list) ]

        self.set_runlist(self._dependency.build_runlist(runlist))
    
    def set_runlist(self, runlist):
        self._runlist = runlist

        for key, cell in self._cells.items():
            if key in self._runlist:
                cell.enable()
            else:
                cell.disable()

    def run(self):
        stopwatch = Stopwatch()
        for key, cell in self._cells.items():
            ready = cell.ready()
            print(f'{stopwatch} [{cell.status}] {cell}')
            if not ready: continue

            try:
                result = cell(dry_run=self.dry_run)
                print(f'{stopwatch} [{cell.status}]')
            except Exception as e:
                print(f'{stopwatch} [{cell.status}] {e}\n')
                self.on_cell_error(cell, e)

    def on_cell_error(self, cell, error):
        if not self.ignore_error: raise error

    def __str__(self):
        lines = [ f"{i:>5}: <[{str(cell.status)}] {str(cell)}>" for i, cell in enumerate(self._cells.values()) ]
        return f'<{self.__name__}\n'+'\n'.join(lines)+'\n>'