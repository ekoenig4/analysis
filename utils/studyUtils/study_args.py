from inspect import signature
import awkward as ak
import re

from ..utils import init_attr
from .variable_table import VariableTable
from ..varConfig import varinfo


def _get_item_from_tree(tree, key):
    if key == 'n_mask':
        return 'n_mask'
    if callable(key):
        return key(tree)
    slice = None
    if any(re.findall(r'\[.*\]', key)):
        slice = re.findall(r'\[.*\]', key)[0]
        key = key.replace(slice, "")
    item = tree[key]
    if slice is not None:
        item = eval(f'item{slice}', {'item': item})
    return item


def _index_item_from_tree(tree, item, index):
    if index is None:
        return item
    if callable(index):
        if len(signature(index).parameters) > 1:
            return index(tree, item)
        index = index(tree)
    return item[index]


def _transform_item_from_tree(tree, item, transform):
    if transform is None:
        return item
    return transform(item)


def _mask_item_from_tree(tree, item, mask):
    if mask is None:
        return item
    if callable(mask):
        # if len(signature(mask).parameters) > 1:
        #     return mask(tree, item)
        mask = mask(tree)

    if item is 'n_mask':
        return ak.sum(mask, axis=-1)

    # if ak.count(item) != ak.count(mask):
    #     return item
    return item[mask]

def _scale_tree(tree, scale):
    if not callable(scale): 
        return scale
    return scale(tree)

def _scale_item_from_tree(tree, item, scale):
    if scale is None:
        return item
    if callable(scale):
        scale = scale(tree)
    return scale*item


class _study_args:
    def __init__(self, treelist, masks=None, indices=None, transforms=None, scale=None, stacked=True, lumi=2018, label=None,
                 saveas=None, return_figax=False, suptitle=None, table=None, tablef=None, report=True, **kwargs):
        if not isinstance(treelist, list):
            treelist = list(treelist)

        kwargs['h_label'] = kwargs.get('h_label', label if label else [
                                       tree.sample for tree in treelist])
        kwargs['h_color'] = kwargs.get(
            'h_color', [tree.color for tree in treelist])
        kwargs['h_systematics'] = kwargs.get(
            'h_systematics', [ tree.systematics for tree in treelist ]
        )

        self.attrs = dict(
            is_data=[tree.is_data for tree in treelist],
            is_signal=[tree.is_signal for tree in treelist],
            is_model=[tree.is_model for tree in treelist],
            stacked=stacked,
            lumi=lumi,
            **kwargs
        )

        self.treelist = treelist
        self.selections = treelist

        ntrees = len(treelist)
        self.masks = init_attr(
            masks, masks if callable(masks) else None, ntrees)
        self.indices = init_attr(
            indices, indices if callable(indices) else None, ntrees)
        self.transforms = init_attr(
            transforms, transforms if callable(transforms) else None, ntrees)

        scales = init_attr(scale, scale if callable(scale) else None, ntrees)
        self.scales = [_scale_tree(tree, scale) for tree, scale in zip(treelist, scales)]

        self.saveas = saveas
        self.return_figax = return_figax
        self.suptitle = suptitle
        self.title = suptitle

        self.report = report

        if table:
            if isinstance(table, bool):
                table = None

            self.tablev = VariableTable(table)
            self.tablef = tablef

    def table(self, var, xlabel, figax, **kwargs):
        if not hasattr(self, 'tablev'): return

        key = xlabel.replace(' ','_')
        info = varinfo.find(var)
        if info is None: info = dict()
        define = info.get('define', None)
        comments = info.get('comments',None)
        extra = self.tablef(**kwargs) if self.tablef else dict()
        
        self.tablev.add(
            key, define, comments, figax=figax, **extra
        )

    def get_histogram(self, key):
        items = [ tree.histograms[key] for tree in self.treelist ]

        counts = [ item.histo for item in items ]
        bins = [ item.bins for item in items ]
        errors = [ item.error for item in items ]

        return counts, bins, errors

    def get_array(self, key, indices=True, transforms=True, scale=False):
        items = [_get_item_from_tree(tree, key) for tree in self.treelist]
        if indices and any(self.indices):
            items = [_index_item_from_tree(tree, item, index) for tree, item, index in zip(
                self.treelist, items, self.indices)]
        if transforms and any(value is not None for value in self.transforms):
            items = [_transform_item_from_tree(tree, item, transform) for tree, item, transform in zip(
                self.treelist, items, self.transforms)]
        if scale and any(value is not None for value in self.scales):
            items = [_scale_item_from_tree(tree, item, scale) for tree, item, scale in zip(self.treelist, items, self.scales)]
        if any(value is not None for value in self.masks):
            items = [_mask_item_from_tree(tree, item, mask) for tree, item, mask in zip(
                self.treelist, items, self.masks)]
        return items

    def get_scale(self, items):
        scales = self.get_array('scale', indices=False, transforms=False, scale=True)
        scales = [
            scale if ak.count(scale) == ak.count(
                item) else ak.ones_like(item)*scale
            for scale, item in zip(scales, items)
        ]
        return scales
