import utils.compat.torch as torch
import numpy as np
import yaml

class Equation:
    @staticmethod
    def numpy_interp(x, xp, fp):
        return np.interp(x, xp, fp)
    
    @staticmethod
    def torch_interp(x, xp, fp):    
        """ From: https://github.com/pytorch/pytorch/issues/50334#issuecomment-1000917964
        One-dimensional linear interpolation for monotonically increasing sample
        points.

        Returns the one-dimensional piecewise linear interpolant to a function with
        given discrete data points :math:`(xp, fp)`, evaluated at :math:`x`.

        Args:
            x: the :math:`x`-coordinates at which to evaluate the interpolated
                values.
            xp: the :math:`x`-coordinates of the data points, must be increasing.
            fp: the :math:`y`-coordinates of the data points, same length as `xp`.

        Returns:
            the interpolated values, same size as `x`.
        """
        xp = xp.to(x.device)
        fp = fp.to(x.device)

        m = (fp[1:] - fp[:-1]) / (xp[1:] - xp[:-1])
        b = fp[:-1] - (m * xp[:-1])

        indicies = torch.sum(torch.ge(x[:, None], xp[None, :]), 1) - 1
        indicies = torch.clamp(indicies, 0, len(m) - 1)

        return m[indicies] * x + b[indicies]
    
    @staticmethod
    def numpy_init(x):
        return np.full_like(x, np.nan)
    
    @staticmethod
    def torch_init(x):
        return torch.full_like(x, torch.nan, device=x.device)

    @staticmethod
    def set_backend(backend):
        assert backend in ['numpy', 'torch'], f'backend must be numpy or torch, not {backend}'

        Equation.backend = backend
        Equation.where = np.where if backend == 'numpy' else torch.where
        Equation.interp = Equation.numpy_interp if backend == 'numpy' else Equation.torch_interp
        Equation.nan = np.nan if backend == 'numpy' else torch.nan
        Equation.array = np.array if backend == 'numpy' else torch.tensor
        Equation.init = Equation.numpy_init if backend == 'numpy' else Equation.torch_init

    def __init__(self, bounds, eq):

        self.bounds = eval(f'lambda x : {bounds}')

        if isinstance(eq, (int, float)):
            self.eq = lambda x : eq

        else:
            eq = Equation.array(eq)
            self.eq = lambda x : Equation.interp(x, eq[:,0], eq[:,1])

    def __call__(self, x):
        mask = self.bounds(x)
        y = self.eq(x)
        return Equation.where(mask, y, Equation.nan)
    
class Piecewise:
    @classmethod
    def from_yaml(cls, filepath):
        with open(filepath, 'r') as f:
            eqs = yaml.safe_load(f)
        return cls.from_dict(eqs)      
    
    @classmethod
    def from_dict(cls, eqs):
        return cls([Equation(bounds, eq) for bounds, eq in eqs.items()])

    def __init__(self, eqs):
        self.eqs = eqs

    def __call__(self, x):
        y = Equation.init(x)
        for eq in self.eqs:
            mask = eq.bounds(x)
            y[mask] = eq.eq(x[mask])
        return y

Equation.set_backend('numpy')
    