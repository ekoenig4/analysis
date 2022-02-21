from datetime import date
import os

from ..utils import *
from ..testUtils import is_iter
from ..varConfig import varinfo
from ..classUtils import TreeIter

date_tag = date.today().strftime("%Y%m%d")


def save_scores(score, saveas):
    directory = f"plots/{date_tag}_plots/scores"
    if not os.path.isdir(directory):
        os.makedirs(directory)
    score.savetex(f"{directory}/{saveas}")


def save_fig(fig, directory, saveas, base=GIT_WD):
    outfn = f"{base}/plots/{date_tag}_plots/{directory}/{saveas}"
    directory = '/'.join(outfn.split('/')[:-1])
    if not os.path.isdir(directory):
        os.makedirs(directory)
    # fig.savefig(f"{directory}/{saveas}.pdf", format="pdf")
    fig.savefig(f"{outfn}.png", format="png", dpi=400)


class Study:
    def __init__(self, selections, labels=None, density=0, log=0, ratio=0, stacked=0, lumikey=2018, sumw2=True, title=None, saveas=None, masks=None, **kwargs):
        if str(type(selections)) == str(TreeIter):
            selections = selections.trees
        elif is_iter(selections):
            selections = list(selections)
        elif type(selections) != list:
            selections = [selections]

        self.selections = selections
        self.masks = masks

        kwargs['s_colors'] = kwargs.get(
            's_colors', [selection.color for selection in selections])
        self.attrs = dict(
            labels=labels if labels else [
                selection.sample for selection in selections],
            is_datas=[selection.is_data for selection in selections],
            is_signals=[selection.is_signal for selection in selections],

            density=density,
            log=log,
            ratio=ratio,
            stacked=stacked,
            lumikey=lumikey,
            sumw2=sumw2,
            **kwargs,
        )

        self.title = title
        self.saveas = saveas

    def get(self, key):
        ie = None
        if ":" in key:
            key, ie = key.split(":")
        items = [selection[key] for selection in self.selections]
        if ie is not None:
            items = [item[:, int(ie)] for item in items]
        if self.masks is not None:
            if type(self.masks) is type(lambda: None):
                items = [item[self.masks(selection)] for item, selection in zip(
                    items, self.selections)]
            else:
                def mask_item(item,mask):
                    if ak.count(item) != ak.count(mask):
                        item = ak.broadcast_arrays(item,mask)[0]
                    return item[mask]
                items = [ mask_item(item,mask) for item, mask in zip(items, self.masks)]
        return items

    def get_scale(self, key):
        hists = self.get(key)
        scales = self.get('scale')
        return [ak.ones_like(hist) * scale for scale, hist in zip(scales, hists)]

    def format_var(self, var, bins=None, xlabel=None):
        info = varinfo.find(var)
        if bins is None and info:
            bins = info.bins
        if info and xlabel is None:
            xlabel = info.get('xlabel',var)
        if xlabel is None: xlabel = var
        return bins, xlabel

    def save_fig(self, fig, directory, base=GIT_WD):
        save_fig(
            fig, directory, self.saveas, base)

    def save_scores(self, scores):
        save_scores(scores, self.saveas)
