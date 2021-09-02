from . import *


date_tag = date.today().strftime("%Y%m%d")

def save_scores(score,saveas):
    directory = f"plots/{date_tag}_plots/scores"
    if not os.path.isdir(directory): os.makedirs(directory)
    score.savetex(f"{directory}/{saveas}")
    
def save_fig(fig,directory,saveas,base=GIT_WD):
    directory = f"{base}/plots/{date_tag}_plots/{directory}"
    if not os.path.isdir(directory): os.makedirs(directory)
    fig.savefig(f"{directory}/{saveas}.pdf",format="pdf")
    
class Study:
    def __init__(self,selections,labels=None,density=0,log=0,ratio=0,stacked=0,lumikey=2018,sumw2=True,title=None,saveas=None,masks=None,varlist=varinfo.keys(),**kwargs):
        if type(selections) == tuple: selections = list(selections)
        if type(selections) != list: selections = [selections]
        
        self.selections = selections
        self.masks = masks

        kwargs['s_colors'] = kwargs.get('s_colors',[ selection.color for selection in selections ])
        self.attrs = dict(
            labels = labels if labels else [ selection.tag for selection in selections ],
            is_datas = [ selection.is_data for selection in selections ],
            is_signals = [ selection.is_signal for selection in selections ],
            
            density = density,
            log = log,
            ratio = ratio,
            stacked = stacked,
            lumikey = lumikey,
            sumw2 = sumw2,
            **kwargs,
        )
        
        
        self.title = title
        self.saveas = saveas
        self.varinfo = {key:varinfo[key] for key in varlist}

    def get(self,key):
        items = [ selection[key] for selection in self.selections ]
        if self.masks is not None:
            items = [ item[mask] for item,mask in zip(items,self.masks) ]
        return items
