try:
    import torch
except ImportError:
    from . import torch

try:
    import tabulate
except ImportError:
    from . import tabulate