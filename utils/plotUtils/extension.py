import matplotlib.pyplot as plt
import numpy as np

from ..classUtils import ObjTransform
from .better_plotter import *

class draw_abcd:
  def __init__(self, x_r, y_r, swapx=False, swapy=False):
    self.x_r, self.y_r = x_r, y_r
    self.swapx, self.swapy = swapx, swapy
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

    if self.swapx:
      regions["A"], regions["B"] = regions["B"], regions["A"]
      regions["C"], regions["D"] = regions["D"], regions["C"]
    if self.swapy:
      regions["A"], regions["C"] = regions["C"], regions["A"]
      regions["B"], regions["D"] = regions["D"], regions["B"]

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

class draw_circle:
  def __init__(self, x, y, r, text=(0,1), color='k', fill=False, **style):
    self.x, self.y, self.r = x, y, r
    self.style = dict(color=color, fill=fill, **style)
    self.tx, self.ty = text
  def __call__(self, fig, ax, histo2d, **kwargs):
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    winw, winh = (xlim[1]-xlim[0]), (ylim[1]-ylim[0])
    circle = plt.Circle((self.x, self.y), self.r, **self.style)  
    ax.add_patch(circle)
    
    mask = ((histo2d.x_array-self.x)/self.r)**2 + ((histo2d.y_array-self.y)/self.r)**2 < 1
    total = histo2d.stats.nevents
    count = np.sum(histo2d.weights[mask])
    eff = count/total

    tx, ty = (self.tx+0.01), (self.ty-0.035)
    ax.text(tx, ty, f'{eff:0.2}', va="center", fontsize=10, transform=ax.transAxes)


class draw_concentric:
  def __init__(self, x, y, r1, r2, text=(0,1), color='k', fill=False, **style):
    self.x, self.y, self.r1, self.r2 = x, y, r1, r2
    self.style = dict(color=color, fill=fill, **style)
    self.tx, self.ty = text
  def __call__(self, fig, ax, histo2d, **kwargs):
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    winw, winh = (xlim[1]-xlim[0]), (ylim[1]-ylim[0])
    inner = plt.Circle((self.x, self.y), self.r1, **self.style)  
    outer = plt.Circle((self.x, self.y), self.r2, **self.style)  
    ax.add_patch(inner)
    ax.add_patch(outer)

    def _get_eff(x, y, r):
      mask = ((histo2d.x_array-x)/r)**2 + ((histo2d.y_array-y)/r)**2 < 1
      total = histo2d.stats.nevents
      count = np.sum(histo2d.weights[mask])
      eff = count/total
      return eff

    inner_eff = _get_eff(self.x, self.y, self.r1)
    outer_eff = _get_eff(self.x, self.y, self.r2) - inner_eff

    tx, ty = (self.tx+0.01), (self.ty-0.035)
    ax.text(tx, ty, f'IN : {inner_eff:0.2}\nOUT: {outer_eff:0.2}', va="center", fontsize=10, transform=ax.transAxes)

from matplotlib.patches import Ellipse
class draw_ellipse:
  def __init__(self, x, y, rx, ry, angle=0, text=(0,1), color='k', fill=False, **style):
    self.x, self.y, self.rx, self.ry, self.angle = x, y, rx, ry, angle
    self.style = dict(color=color, fill=fill, **style)
    self.tx, self.ty = text
  def __call__(self, fig, ax, histo2d, **kwargs):
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    winw, winh = (xlim[1]-xlim[0]), (ylim[1]-ylim[0])
    circle = Ellipse((self.x, self.y), 2*self.rx, 2*self.ry, angle=self.angle, **self.style)  
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
    ax.text(tx, ty, f'{eff:0.2}', va="center", fontsize=10, transform=ax.transAxes)

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
