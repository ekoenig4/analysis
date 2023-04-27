import pkgutil, importlib

__all__ = []

def register_modules(__all__=__all__):
    """ Register all modules in this package """
    for loader, module_name, is_pkg in pkgutil.walk_packages(__path__):
        module = importlib.import_module(f'.{module_name}', package=__name__)
        globals()[module_name] = getattr(module, module_name)
        __all__.append(module_name)