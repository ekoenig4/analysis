# from .plotUtils import *
from .extension import *
from .better_plotter import *
from .better_plotter_2d import *
from .multi_plotter import hist_multi, count_multi
from .multi_plotter_2d import hist2d_multi, hist2d_simple
from .histogram import *
from .histogram2d import *
from .graph import *
from .model import *
from . import function

plt.style.use(['science','no-latex'])
plt.rcParams["figure.figsize"] = (6.5,6.5)
plt.rcParams['font.size'] =  11