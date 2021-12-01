from ..plotUtils import *
from ..studyUtils import *


def plot_history(history,metric='loss'):
    loss = history.history[metric]
    val_loss = history.history[f'val_{metric}']

    fig,ax = plt.subplots(figsize=(8,5))
    graph_multi(np.arange(1,len(loss)+1),[loss,val_loss],labels=["Training","Validation"],xlabel="Epoch",ylabel=metric.capitalize(),figax=(fig,ax))
    return fig,ax

def plot_efficiency(nfound,nactual,nobj=6):
    obj = {6:'Signal Jets',3:'Higgs'}.get(nobj)
    eff = safe_divide(nfound,nactual,1)
    fig,axs = plt.subplots(nrows=1,ncols=2,figsize=(16,5))
    hist_multi([nactual,nfound],bins=range(nobj+2),labels=["Actual","Found"],xlabel=f"N {obj}",figax=(fig,axs[0]))
    hist2d_simple(nactual,nfound,xbins=range(nobj+2),ybins=range(nobj+2),labels=["Actual","Found"],xlabel=f"N Actual {obj}",ylabel=f"N Found {obj}",log=1,figax=(fig,axs[1]))
    fig.suptitle(f"Efficieny: {ak.mean(eff):0.3}")
    fig.tight_layout()
    return fig

def compare_methods(nmethod1,nmethod2,method1='Method 1',method2='Method 2',nobj=6):
    fig,axs = plt.subplots(nrows=1,ncols=2,figsize=(16,5))
    hist_multi([nmethod1,nmethod2],bins=range(8),ratio=1,labels=[method1,method2],figax=(fig,axs[0]))
    hist2d_simple(nmethod1,nmethod2,xbins=range(8),ybins=range(8),xlabel=f"{method1} Signal Jets",ylabel=f"{method2} Signal Jets",figax=(fig,axs[1]))
    fig.tight_layout()
    return fig

def plot_features(features):
    nrows,ncols = study.autodim(len(features.fields),flip=True)

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(int((16/3)*ncols), 5*nrows))
    for i, field in enumerate(features.fields):
        hist_multi([features[field]],
                xlabel=field, figax=(fig, axs.flat[i]))
    fig.tight_layout()

def plot_pairplot(features):
    nrows,ncols = len(features.fields),len(features.fields)
    fig,axs = plt.subplots(nrows=nrows,ncols=ncols,figsize=(int((16/3)*ncols), 5*nrows))

    for i , ifield in enumerate(features.fields):
        for j , jfield in enumerate(features.fields):
            if i == j:
                hist_multi([features[ifield]],xlabel=ifield,ylabel=jfield,figax=(fig,axs[i,j]))
            else:
                hist2d_simple(features[ifield],features[jfield],xlabel=ifield,ylabel=jfield,figax=(fig,axs[i,j]))
    fig.tight_layout()
    return fig
