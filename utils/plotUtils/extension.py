from matplotlib.patches import Ellipse
from matplotlib import patheffects
import matplotlib.pyplot as plt
import numpy as np

from ..classUtils import ObjTransform, ObjIter
from .better_plotter import *

class obj_store:
    def __init__(self):
        self.objs = []
    def __getitem__(self, key): return self.objs[key]
    def __iter__(self): return iter(self.objs)
    def __repr__(self): return repr(self.objs)
    def append(self, objs):
        self.objs.append(objs)

class draw_line:
    def __init__(self, x):
        self.x = x

    def __call__(self, fig, ax, **kwargs):
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        ax.plot([self.x, self.x], ylim, color='grey', linestyle=':')

class draw_cut:
    def __init__(self, x, selection='>='):
        self.x = x
        self.selection = selection

    def _array_eff(self, histo):
        array = histo.array
        weights = histo.weights 
        total = np.sum(weights)
        local = dict()
        exec(f'mask = array {self.selection} {self.x}',dict(array=array),local)
        count = np.sum(weights[local['mask']])
        eff = count/total 
        return f'{eff:0.2%}'

    def _count_eff(self, histo):
        array = histo.bins[:-1]
        weights = histo.histo 
        total = np.sum(weights)
        local = dict()
        exec(f'mask = array {self.selection} {self.x}',dict(array=array),local)
        count = np.sum(weights[local['mask']])
        eff = count/total 
        return f'{eff:0.2%}'

    def get_eff(self, histo):
        if getattr(histo, 'array', None) is not None: return self._array_eff(histo)
        return self._count_eff(histo)

    def __call__(self, fig, ax, histos=None, stack=None, **kwargs):
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        ax.plot([self.x, self.x], ylim, color='grey', linestyle=':')


class draw_abcd:
    def __init__(self, x_r, y_r, regions=["A","B","C","D"]):
        self.x_r, self.y_r = x_r, y_r
        self.regions=regions
        self.store = []

    def __call__(self, fig, ax, histo2d, **kwargs):
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        winw, winh = (xlim[1]-xlim[0]), (ylim[1]-ylim[0])

        x_lo, x_mi, x_hi = self.x_r
        y_lo, y_mi, y_hi = self.y_r

        style = dict(fill=False, ec='k')
        regions = [
            plt.Rectangle((x_mi, y_mi), x_hi-x_mi, y_hi-y_mi, **style),
            plt.Rectangle((x_lo, y_mi), x_mi-x_lo, y_hi-y_mi, **style),
            plt.Rectangle((x_mi, y_lo), x_hi-x_mi, y_mi-y_lo, **style),
            plt.Rectangle((x_lo, y_lo), x_mi-x_lo, y_mi-y_lo, **style),
        ]
        regions = { region:box for region, box in zip(self.regions, regions) }
        nevents = histo2d.stats.nevents

        def _get_eff(obj):
            x, y = obj.get_xy()
            h, w = obj.get_height(), obj.get_width()
            x_mask = (histo2d.x_array >= x) & (histo2d.x_array < x + w)
            y_mask = (histo2d.y_array >= y) & (histo2d.y_array < y + h)

            count = np.sum(histo2d.weights[x_mask & y_mask])
            return count, count/nevents

        r_info = {}
        for r, obj in regions.items():
            yields, eff = _get_eff(obj)
            r_info[r] = {'yields':yields,'eff':eff}
            x, y = obj.get_xy()
            h, w = obj.get_height(), obj.get_width()

            tx = x + 0.01*winw
            if tx > x+w:
                tx = x + w/2

            ty = y + h - 0.035*winh
            if ty < y:
                ty = y + h/2
            txt = ax.text(tx, ty, r, va="center", fontsize=12)
            txt.set_path_effects(
                [patheffects.withStroke(linewidth=2, foreground='w')])
            ax.add_patch(obj)

        region_total = sum( info['yields'] for info in r_info.values() )
        sr_total = sum( r_info[r]['yields'] for r in ('A','B') )
        cr_total = sum( r_info[r]['yields'] for r in ('C','D') )
        tr_total = sum( r_info[r]['yields'] for r in ('A','C') )
        er_total = sum( r_info[r]['yields'] for r in ('B','D') )

        lines = [
        f"Total: {region_total:0.2e} ({region_total/nevents:0.2%})",
        f"SR   : {sr_total:0.2e} ({sr_total/nevents:0.2%})",
        f"CR   : {cr_total:0.2e} ({cr_total/nevents:0.2%})",
        f"TR   : {tr_total:0.2e} ({tr_total/nevents:0.2%})",
        f"ER   : {er_total:0.2e} ({er_total/nevents:0.2%})",
        ] 

        lines2 = [
            f"{r}    : {r_info[r]['yields']:0.2e} ({r_info[r]['eff']:0.2%})"
            for r in ("A","B","C","D")
        ]

        max_columns = max( len(line) for line in lines+lines2 )
        lines = [ line.ljust(max_columns) for line in lines + ['-'*max_columns] + lines2 ]

        label = '\n'.join(lines)

        txt = ax.text( 0.9, 1.05, label, ha="center", va="top", fontsize=10, transform=ax.transAxes, color='w' )
        txt.set_bbox(dict(facecolor='k', alpha=0.75, edgecolor='k'))
        self.store.append(r_info)


class draw_circle:
    def __init__(self, x, y, r, efficiency=False, label=None, text=(0, 1), color='k', fill=False, **style):
        self.x, self.y, self.r = x, y, r
        self.style = dict(color=color, fill=fill, **style)
        self.tx, self.ty = text or (None, None)
        self.efficiency = efficiency
        self.label = label

    def __call__(self, fig, ax, histo2d, **kwargs):
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        winw, winh = (xlim[1]-xlim[0]), (ylim[1]-ylim[0])
        circle = plt.Circle((self.x, self.y), self.r, **self.style)
        ax.add_patch(circle)

        mask = ((histo2d.x_array-self.x)/self.r)**2 + \
            ((histo2d.y_array-self.y)/self.r)**2 < 1
        total = histo2d.stats.nevents
        count = np.sum(histo2d.weights[mask])
        eff = count/total

        if self.label is None:
            label = f"{count:0.2e}({eff:0.2%})"
        else:
            label = self.label.format(**locals())

        ax.set_aspect('auto')
        ax.set_xlim(xlim)

        if None in (self.tx, self.ty): return
        
        tx, ty = (self.tx+0.01), (self.ty-0.035)
        txt = ax.text(tx, ty, label,
                va="center", fontsize=10, transform=ax.transAxes, color='w')
        # txt.set_path_effects(
        #         [patheffects.withStroke(linewidth=2, foreground='w')])
        txt.set_bbox(dict(facecolor=self.style['color'], alpha=0.75, edgecolor=self.style['color']))


class draw_concentric:
    def __init__(self, x, y, r1, r2, efficiency=False, label=None, text=(0, 0.95), textstyle=dict(), color='k', fill=False, **style):
        self.x, self.y, self.r1, self.r2 = x, y, r1, r2
        self.style = dict(color=color, fill=fill, **style)
        self.textstyle = textstyle
        self.tx, self.ty = text
        self.label = label
        self.efficiency = efficiency

    def __call__(self, fig, ax, histo2d, **kwargs):
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        winw, winh = (xlim[1]-xlim[0]), (ylim[1]-ylim[0])
        inner = plt.Circle((self.x, self.y), self.r1, **self.style)
        outer = plt.Circle((self.x, self.y), self.r2, **self.style)
        ax.add_patch(inner)
        ax.add_patch(outer)

        total = histo2d.stats.nevents 
        def _get_count(x, y, r):
            mask = ((histo2d.x_array-x)/r)**2 + ((histo2d.y_array-y)/r)**2 < 1
            count = np.sum(histo2d.weights[mask])
            return count

        total_count = _get_count(self.x, self.y, self.r2)
        inner_count = _get_count(self.x, self.y, self.r1)
        outer_count = total_count - inner_count

        total_eff = total_count/total 
        inner_eff = inner_count/total 
        outer_eff = outer_count/total

        
        if self.label is None:
            label = [
                f"Total: {total_count:0.2e} ({total_eff:0.2%})",
                f"SR   : {inner_count:0.2e} ({inner_eff:0.2%})",
                f"CR   : {outer_count:0.2e} ({outer_eff:0.2%})",
                ] 
            label = '\n'.join(label)
        else:
            label = self.label.format(**locals())


        tx, ty = (self.tx+0.01), (self.ty-0.035)
        txt = ax.text(tx, ty, label,
                va="center", fontsize=10, transform=ax.transAxes, color='w')
        # txt.set_path_effects(
        #         [patheffects.withStroke(linewidth=2, foreground='w')])
        txt.set_bbox(dict(facecolor=self.style['color'], alpha=0.75, edgecolor=self.style['color']))


class draw_ellipse:
    def __init__(self, x, y, rx, ry, angle=0, text=(0, 1), color='k', fill=False, **style):
        self.x, self.y, self.rx, self.ry, self.angle = x, y, rx, ry, angle
        self.style = dict(color=color, fill=fill, **style)
        self.tx, self.ty = text

    def __call__(self, fig, ax, histo2d, **kwargs):
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        winw, winh = (xlim[1]-xlim[0]), (ylim[1]-ylim[0])
        circle = Ellipse((self.x, self.y), 2*self.rx, 2 *
                         self.ry, angle=self.angle, **self.style)
        ax.add_patch(circle)

        rad = np.deg2rad(self.angle)
        dx, dy = histo2d.x_array - self.x, histo2d.y_array - self.y
        x = np.cos(rad)*dx + np.sin(rad)*dy
        y = np.cos(rad)*dy - np.sin(rad)*dx
        r2 = (x/self.rx)**2 + (y/self.ry)**2
        mask = r2 < 1

        total = histo2d.stats.nevents
        count = np.sum(histo2d.weights[mask])
        eff = count/total

        tx, ty = (self.tx+0.01), (self.ty-0.035)
        ax.text(tx, ty, f'{eff:0.2}', va="center",
                fontsize=10, transform=ax.transAxes)


def plot_histo2d_x_corr(histo2d, fig, ax, **kwargs):
    corr = histo2d.x_corr(marker=None)
    # ax.text(0.1, 0.1, f'slope = {corr.fit.c1:0.2}', transform=ax.transAxes)
    plot_graph(corr, figax=(fig, ax), fill_error=True,
               xlim=ax.get_xlim(), ylim=ax.get_ylim())


def plot_histo2d_y_corr(histo2d, fig, ax, **kwargs):
    corr = histo2d.y_corr(marker=None)
    # ax.text(0.1, 0.1, f'slope = {corr.fit.c1:0.2}', transform=ax.transAxes)
    plot_graph(corr, figax=(fig, ax), fill_error=True,
               xlim=ax.get_xlim(), ylim=ax.get_ylim())


def plot_histo2d_xy_corr(histo2d, fig, ax, **kwargs):
    plot_histo2d_x_corr(histo2d, fig, ax, **kwargs)
    plot_histo2d_y_corr(histo2d, fig, ax, **kwargs)
