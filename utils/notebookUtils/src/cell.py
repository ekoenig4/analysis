import inspect

try:
    from termcolor import colored
except ModuleNotFoundError:
    def colored(text, *args, **kwargs):
        return text

class Status:
    def __init__(self, status, color, attrs=[]):
        self.status = status
        self.color = color
        self.attrs = attrs

    def __str__(self):
        return colored(self.status.center(8), self.color, attrs=self.attrs)

    def __repr__(self):
        return f"status={self.status}"
    
    @classmethod
    @property
    def pending(cls):
        return cls('pending', 'blue')
    
    @property
    def is_pending(self):
        return self.status == 'pending'

    @classmethod
    @property
    def running(cls):
        return cls('running', 'yellow')

    @classmethod
    @property
    def done(cls):
        return cls('done', 'green')
    
    @property
    def is_done(self):
        return self.status == 'done'
    
    @classmethod
    @property
    def failed(cls):
        return cls('failed', 'red')

    @classmethod
    @property
    def disabled(cls):
        return cls('disabled', 'white', attrs=['dark'])
    
    @property
    def is_disabled(self):
        return self.status == 'disabled'

class Cell:
    """
    A cell in a notebook. Similar to a cell in a jupyter notebook.
    """
    
    def __init__(self, notebook, method):
        self.notebook = notebook
        self.method = method
        self.__name__ = method.__name__
        self.status = Status.pending

        self._build_args()

    def pending(self):
        self.status = Status.pending
    def enable(self): 
        if self.status.is_done: return
        self.status = Status.pending
    def disable(self): 
        if self.status.is_done: return
        self.status = Status.disabled
    def ready(self): 
        _ready = self.status.is_pending
        if _ready: self.status = Status.running
        return _ready

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

        if self.use_kwargs:
            self._run = self._run_with_kwargs
        else:
            self._run = self._run_with_args

    def _run_with_kwargs(self):
        namespace = self.notebook.namespace
        return self.method(self.notebook, **namespace)
    
    def _run_with_args(self):
        namespace = self.notebook.namespace
        return self.method(self.notebook, *[ namespace.get( key, None) for key in self.args ])

    def __call__(self, dry_run=False):
        result = None
        try:
            result = self._run() if not dry_run else None
        except Exception as e:
            self.status = Status.failed
            raise e

        self.status = Status.done
        return result

    def __str__(self):
        name = colored(self.__name__, 'white', attrs=['dark'] if self.status.is_disabled else [])
        args = colored(', '.join(self.args), 'white', attrs=['dark'])
        return f"{name}({args})"

    def __repr__(self):
        return str(self)
