import warnings
warnings.filterwarnings("ignore")

import numpy as np

def group_kwargs(prefix, **kwargs):
    grouped_kwargs = { key[len(prefix):]:value for key,value in kwargs.items() if key.startswith(prefix) }
    remaining_kwargs = { key:value for key,value in kwargs.items() if not key.startswith(prefix) }
    return grouped_kwargs, remaining_kwargs

def _set_text(ax, text=None, text_style={}, **kwargs):
    if text is not None:
        style = dict(transform=ax.transAxes, va="center")
        style.update(**text_style)
        if isinstance(text, tuple): text = [text]
        for x, y, s in text:
            ax.text(x, y, s, **style)
    return kwargs

def _set_rescale(ax, **kwargs):
    ax.autoscale()
    return kwargs

def _set_legend(ax, legend=False, legend_loc='upper left', is_2d=False, **kwargs):
    legend_kwargs, kwargs = group_kwargs('legend_', **kwargs)

    if legend: 
        ax.legend(loc=legend_loc, **legend_kwargs)
    return kwargs

def _get_ylabel(density=False, cumulative=False, efficiency=False, scale=None):
    ylabel = "Events"
    if scale == "xs":
        ylabel = "Cross-Section (pb)"
    if efficiency:
        ylabel = "PDF"
    if density:
        ylabel = "PDF"
    if cumulative or cumulative == 1:
        ylabel = "CDF Below"
    if cumulative == -1:
        ylabel = "CDF Above"
    return ylabel

def _set_ylabel(ax, density=False, cumulative=False, efficiency=False, scale=None, **kwargs):
    kwargs['ylabel'] = kwargs.get('ylabel',_get_ylabel(density,cumulative,efficiency, scale))
    return kwargs

def _get_ylim(ax, yscale=None, log=False, is_2d=False):
    if is_2d: return ax.get_ylim()
    ymin,ymax = ax.get_ylim()
    if yscale is None:
        yscale = (0.1,100) if log else (0,1.5)
    ymin_scale,ymax_scale = yscale
    return ( min(ymax_scale*ymin,ymin_scale*ymin), max(ymin_scale*ymax,ymax_scale*ymax) )

def _set_ylim(ax, yscale=None, log=False, logy=False, is_2d=False, **kwargs):
    ylim = kwargs.get('ylim',_get_ylim(ax, yscale, log or logy, is_2d))
    if ylim is None: ylim = ax.get_ylim()
    
    ylo, yhi = ylim
    if ylo is None: ylo = ax.get_ylim()[0]
    if yhi is None: yhi = ax.get_ylim()[1]

    kwargs['ylim'] = (ylo, yhi)
    if log or logy: ax.set_yscale('log')
    return kwargs

def _set_xlabel(ax, xlabel=None, **kwargs):
    if not isinstance(xlabel,list):
        kwargs['xlabel'] = xlabel 
        return kwargs 
    
    ax.set_xticks(np.arange(len(xlabel))+0.5)
    rotation = -45 if isinstance(xlabel[0],str) else 0
    ax.set_xticklabels(xlabel, rotation=rotation)
    return kwargs

def _set_xlim(ax, logx=False, **kwargs):
    if logx: ax.set_xscale('log')

    return kwargs

def _set_grid(ax, grid=False, **kwargs):
    if grid: ax.grid()
    return kwargs

def _set_lumi(ax, lumi=None, is_2d=False, **kwargs):
    if is_2d: return kwargs 
    
    from ..xsecUtils import lumiMap
    
    lumi,label = lumiMap.get(lumi,(lumi,None))
    
    if label is None: return kwargs 
    
    text = f"{lumi/1000:0.1f} $fb^{'{-1}'}$ {label}"
    _set_text(ax, (1.0, 1.0, text), text_style=dict(ha="right", va="bottom", fontsize=10))
    
    return kwargs

_set_axs = [ func for key,func in vars().items() if key.startswith('_set_') and callable(func) ]

def format_axes(ax,is_2d=False, **kwargs):
    for set_func in _set_axs: 
        kwargs = set_func(ax, is_2d=is_2d, **kwargs)
        if 'is_2d' in kwargs: kwargs.pop('is_2d')

    if any(kwargs): ax.set(**kwargs)