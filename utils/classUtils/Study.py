from datetime import date
import os
import re
from inspect import signature

from ..utils import *
from ..testUtils import is_iter
from ..varConfig import varinfo
from ..classUtils import TreeIter, ObjIter

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

    fig.savefig(f"{outfn}.pdf", format="pdf")
    fig.savefig(f"{outfn}.png", format="png", dpi=400)
    # fig.savefig(f"{outfn}.png", format="png")


def format_var(var, bins=None, xlabel=None):
    info = varinfo.find(var)
    if info and bins is None:
        bins = info.get('bins', None)
    if info and xlabel is None:
        xlabel = info.get('xlabel', var)
    if xlabel is None:
        xlabel = var

    if isinstance(xlabel, str) and any(re.findall(r'\[.*\]', xlabel)):
        slice = re.findall(r'\[.*\]', xlabel)[0]
        place = int(next(idx for idx in slice[1:-1].split(',') if idx != ':'))
        xlabel = f'{ordinal(place+1)} {xlabel.replace(slice,"")}'

    return bins, xlabel


def _scale_items(self, items):
    def scale_item(item, selection, mask):
        if mask is None:
            return item
        if callable(mask):
            mask = mask(selection)
        if isinstance(mask, str):
            mask = selection[mask]
        return mask*item

    if callable(self.scales):
        items = [scale_item(item, selection, self.scales) for item, selection in zip(
            items, self.selections)]
    else:
        items = [scale_item(item, selection, mask) for item, selection, mask in zip(
            items, self.selections, self.scales)]
    return items


def _index_items(self, items):
    def index_item(item, selection, mask):
        if mask is None:
            return item
        if callable(mask):
            sign = signature(mask)
            if len(sign.parameters) == 1:
                mask = mask(selection)
            else:
                return mask(selection, item)

        return item[mask]

    if callable(self.indicies):
        items = [index_item(item, selection, self.indicies) for item, selection in zip(
            items, self.selections)]
    else:
        items = [index_item(item, selection, mask) for item, selection, mask in zip(
            items, self.selections, self.indicies)]
    return items


def _mask_items(self, items):
    def mask_item(item, selection, mask):
        if mask is None:
            return item
        if callable(mask):
            mask = mask(selection)
        if item is 'n_mask':
            item = ak.sum(mask, axis=-1)

        if ak.count(item) == ak.count(mask):
            return item[mask]
        return item

    if callable(self.masks):
        items = [mask_item(item, selection, self.masks) for item, selection in zip(
            items, self.selections)]
    else:
        items = [mask_item(item, selection, mask) for item, selection, mask in zip(
            items, self.selections, self.masks)]
    return items


def _transform_items(self, items):
    if callable(self.transforms):
        items = [self.transforms(item) for item, selection in zip(
            items, self.selections)]
    else:
        items = [transform(item) if transform is not None else item for item,
                 transform in zip(items, self.transforms)]
    return items


class Study:
    def __init__(self, selections, label=None, density=0, log=0, ratio=0, stacked=0, lumi=2018, sumw2=True, title=None,
                 saveas=None, masks=None, transforms=None, indicies=None, return_figax=False, return_store=False, scale=None, **kwargs):
        if str(type(selections)) == str(TreeIter):
            selections = selections.trees
        elif str(type(selections)) == str(ObjIter):
            selections = selections.objs
        elif is_iter(selections):
            selections = list(selections)
        elif type(selections) != list:
            selections = [selections]

        self.selections = selections
        self.masks = masks
        self.transforms = transforms
        self.indicies = indicies
        self.scales = scale

        kwargs['h_color'] = kwargs.get(
            'h_color', [selection.color for selection in selections])
        self.attrs = dict(
            h_label=label if label else [
                selection.sample for selection in selections],
            is_data=[selection.is_data for selection in selections],
            is_signal=[selection.is_signal for selection in selections],
            is_model=[selection.is_model for selection in selections],
            density=density,
            log=log,
            ratio=ratio,
            stacked=stacked,
            lumi=lumi,
            h_sumw2=sumw2,
            **kwargs,
        )

        self.title = title
        self.saveas = saveas
        self.return_figax = return_figax
        self.return_store = return_store

    def get(self, key, transform=True, indicies=True, scale=False):
        def _get_item(selection, key, ie=None):
            if key == 'n_mask':
                return 'n_mask'
            if callable(key):
                return key(selection)
            slice = None
            if any(re.findall(r'\[.*\]', key)):
                slice = re.findall(r'\[.*\]', key)[0]
                key = key.replace(slice, "")
            item = selection[key]
            if slice is not None:
                item = eval(f'item{slice}', {'item': item})
            return item
        items = [_get_item(selection, key) for selection in self.selections]
        if scale and self.scales is not None:
            items = _scale_items(self, items)
        if indicies and self.indicies is not None:
            items = _index_items(self, items)
        if self.masks is not None:
            items = _mask_items(self, items)
        if transform and self.transforms is not None:
            items = _transform_items(self, items)
        return items

    def get_scale(self, hists):
        scales = self.get('scale', transform=False, indicies=False, scale=True)
        scales = [ak.ones_like(hist) * scale for scale,
                  hist in zip(scales, hists)]
        # scales =  [scale for scale, hist in zip(scales, hists)]
        return scales

    def format_var(self, var, bins=None, xlabel=None):
        return format_var(var, bins, xlabel)

    def save_fig(self, fig, directory, base=GIT_WD):
        save_fig(
            fig, directory, self.saveas, base)

    def save_scores(self, scores):
        save_scores(scores, self.saveas)
