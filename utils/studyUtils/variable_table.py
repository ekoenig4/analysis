import os
from ..config import GIT_WD
from ..classUtils import ObjIter
from ..ak_tools import get_avg_std
from ..plotUtils import obj_store

from ..studyUtils import default_args 

csv = None

def add_auroc(e_c_store=None, h_transform=None, **kwargs):
    auroc_mean, auroc_stdv = get_avg_std(ObjIter(e_c_store[-1]).stats.area.npy)
    return dict(
        transform=str(h_transform),
        auroc_mean=auroc_mean,
        auroc_stdv=auroc_stdv
    )

auroc_args = dict(
    default_args.auroc,
    tablef=add_auroc
)


class VariableTable:
    def __init__(self, fname=None, base='logs/'):
        import pandas as pd

        if csv: fname = csv

        assert fname is not None, "Please specify a filename for the variable table"

        fname = os.path.join(GIT_WD, base, fname)
        if not fname.endswith('.csv'): fname += '.csv'
        
        self.fname = fname
        self.plots = os.path.join(base, fname).replace('.csv','/')

        if os.path.exists(fname):
            self.table = pd.read_csv(fname)
        else:
            self.table = pd.DataFrame()

        self.header = []
        
    def add(self, variable, define=None, comments=None, figax=None, **kwargs):
        import pandas as pd

        fig, ax = figax

        from .studyUtils import save_fig
        save_fig(fig, self.plots, variable, fmt='png')
        
        self.header = ['variable','define','comments'] + list(kwargs.keys()) + ['plot']
        row = [variable, define, comments] + list(kwargs.values()) + [f'{variable}.png']
        table = pd.DataFrame([row], columns=self.header)
        
        if any(self.table):
            self.table = self.table[self.table['variable'] != variable]

        self.table = pd.concat([self.table, table], ignore_index=True)
        self.save()

    def save(self):
        self.table.to_csv(self.fname, index=False)

    def __repr__(self): return repr(self.table)

