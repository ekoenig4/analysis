import warnings
warnings.filterwarnings("ignore")

def _set_rescale(ax, **kwargs):
    ax.autoscale()
    return kwargs

def _set_legend(ax, legend=False, is_2d=False, **kwargs):
    if legend: ax.legend()
    return kwargs

def _get_ylabel(density=False, cumulative=False, scale=None):
    ylabel = "Events"
    if scale == "xs":
        ylabel = "Cross-Section (pb)"
    if density:
        ylabel = "PDF"
    if cumulative or cumulative == 1:
        ylabel = "CDF Below"
    if cumulative == -1:
        ylabel = "CDF Above"
    return ylabel

def _set_ylabel(ax, density=False, cumulative=False, scale=None, **kwargs):
    kwargs['ylabel'] = kwargs.get('ylabel',_get_ylabel(density,cumulative,scale))
    return kwargs

def _get_ylim(ax, yscale=None, log=False, is_2d=False):
    if is_2d: return ax.get_ylim()
    ymin,ymax = ax.get_ylim()
    if yscale is None:
        yscale = (0.1,10) if log else (0,1.5)
    ymin_scale,ymax_scale = yscale
    return ( min(ymax_scale*ymin,ymin_scale*ymin), max(ymin_scale*ymax,ymax_scale*ymax) )

def _set_ylim(ax, yscale=None, log=False, is_2d=False, **kwargs):
    kwargs['ylim'] = kwargs.get('ylim',_get_ylim(ax, yscale, log, is_2d))
    if log: ax.set_yscale('log')
    return kwargs

def _set_xlabel(ax, xlabel=None, **kwargs):
    if not isinstance(xlabel,list):
        kwargs['xlabel'] = xlabel 
        return kwargs 
    
    ax.set_xticks(range(len(xlabel)))
    rotation = -45 if isinstance(xlabel[0],str) else 0
    ax.set_xticklabels(xlabel, rotation=rotation)
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
    ax.text(0.75,1.04, text,ha="center", va="center", transform=ax.transAxes)
    
    return kwargs

_set_axs = [ func for key,func in vars().items() if key.startswith('_set_') and callable(func) ]

def format_axes(ax,is_2d=False, **kwargs):
    for set_func in _set_axs: 
        kwargs = set_func(ax, is_2d=is_2d, **kwargs)
        if 'is_2d' in kwargs: kwargs.pop('is_2d')
    
    if any(kwargs): ax.set(**kwargs)