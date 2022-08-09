import matplotlib.pyplot as plt
import numpy as np

from ..classUtils import ObjTransform
from .better_plotter import *

class draw_abcd(ObjTransform):
  def __call__(self, fig, ax, histo2d, **kwargs):
    xlim, ylim = ax.get_xlim(), ax.get_ylim()

    winw, winh = (xlim[1]-xlim[0]), (ylim[1]-ylim[0])

    x_lo, x_mi, x_hi = self.x_r
    y_lo, y_mi, y_hi = self.y_r

    style = dict(fill=False, ec='k')
    regions = {
      "A":plt.Rectangle((x_mi, y_mi), x_hi-x_mi, y_hi-y_mi, **style),
      "B":plt.Rectangle((x_lo, y_mi), x_mi-x_lo, y_hi-y_mi, **style),
      "C":plt.Rectangle((x_mi, y_lo), x_hi-x_mi, y_mi-y_lo, **style),
      "D":plt.Rectangle((x_lo, y_lo), x_mi-x_lo, y_mi-y_lo, **style),
    }

    def _get_eff(obj):
      x, y = obj.get_xy()
      h, w = obj.get_height(), obj.get_width()
      x_mask = (histo2d.x_array >= x) & (histo2d.x_array < x + w)
      y_mask = (histo2d.y_array >= y) & (histo2d.y_array < y + h)

      total = histo2d.stats.nevents
      count = np.sum(histo2d.weights[x_mask & y_mask])
      return count/total

    for r, obj in regions.items(): 
      eff = _get_eff(obj)
      x, y = obj.get_xy()
      h, w = obj.get_height(), obj.get_width()

      tx = x + 0.01*winw
      if tx > x+w: tx = x + w/2

      ty = y + h - 0.035*winh
      if ty < y: ty = y + h/2

      ax.text(tx, ty, f'{r}: {eff:0.2}', va="center", fontsize=10)
      ax.add_patch(obj)

def plot_histo2d_x_corr(histo2d, fig, ax, **kwargs):
  corr = histo2d.x_corr(marker=None, fit='linear')
  ax.text(0.1,0.1,f'slope = {corr.fit.c1:0.2}',transform=ax.transAxes)
  plot_graph(corr, figax=(fig,ax), fill_error=True, xlim=ax.get_xlim(), ylim=ax.get_ylim())

def plot_histo2d_y_corr(histo2d, fig, ax, **kwargs):
  corr = histo2d.y_corr(marker=None, fit='linear')
  ax.text(0.1,0.1,f'slope = {corr.fit.c1:0.2}',transform=ax.transAxes)
  plot_graph(corr, figax=(fig,ax), fill_error=True, xlim=ax.get_xlim(), ylim=ax.get_ylim())

def plot_histo2d_xy_corr(histo2d, fig, ax, **kwargs):
  plot_histo2d_x_corr(histo2d, fig, ax, **kwargs)
  plot_histo2d_y_corr(histo2d, fig, ax, **kwargs)
