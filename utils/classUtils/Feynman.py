import networkx as nx
from itertools import permutations, product
from collections import defaultdict
from functools import reduce 
import numpy as np
import pandas as pd 


def generation_position(graph):
    posMap=nx.get_node_attributes(graph,'pos')
    gen_pos = defaultdict(list)
    for gen, y in posMap.values():
        gen_pos[gen].append(y)
    gen_shifts = { gen: 1-np.mean(ys)  for gen, ys in gen_pos.items() }
    posMap = { label : (x, y+gen_shifts[x]) for label, (x,y) in posMap.items() }
    return posMap

class Feynman:
    def __init__(self, typeid : str):
        self.id = 0
        self.typeid = typeid
        self.products = []
    def decays(self, *products):
        self.products = [ product if isinstance(product, Feynman) else Feynman(product) for product in products ]
        for product in self.products:
            product.mother = self
        return self

    def get_finalstate(self):
        if getattr(self, 'finalstate', None): return self.finalstate

        def _finalstate(feynman):
            if not any(feynman.products):
                return [feynman]

            finalstates = []
            for product in feynman.products:
                finalstates += _finalstate(product)
            return finalstates
        self.finalstate = _finalstate(self)
        return self.finalstate

    def get_finalstate_types(self):
        if getattr(self, 'finalstate_types', None): return self.finalstate_types

        finalstate = self.get_finalstate()
        self.finalstate_type = defaultdict(list)
        for state in finalstate:
            self.finalstate_type[state.typeid].append(state)
        return self.finalstate_type

    def get_internalstate(self):
        if getattr(self, 'internalstate', None): return self.internalstate

        def _internalstate(feynman):
            if not any(feynman.products):
                return []

            internalstates = []
            for product in feynman.products:
                internalstates += _internalstate(product)
            internalstates += [feynman]
            return internalstates
        self.internalstate = _internalstate(self)
        return self.internalstate

    def get_internalstate_types(self):
        if getattr(self, 'internalstate_types', None): return self.internalstate_types

        internalstate = self.get_internalstate()
        self.internalstate_type = defaultdict(list)
        for state in internalstate:
            self.internalstate_type[state.typeid].append(state)
        return self.internalstate_type
        

    def build_diagram(self):
        if getattr(self, 'diagram', None): return self.diagram

        _multi = defaultdict(lambda:0)
        _colors = dict()
        def _build_diagram(feynman, generation=0, pos=(0,0)):
            feynman.generation=generation
            _multi[feynman.typeid] += 1
            _multi[generation] += 1
            if feynman.typeid not in _colors:
                _colors[feynman.typeid] = len(_colors)

            diagram = nx.DiGraph()

            pos = (generation, -_multi[generation])
            diagram.add_node(f'{feynman.typeid}{_multi[feynman.typeid]}', pos=pos, color=_colors[feynman.typeid])

            if any(feynman.products):
                for product in feynman.products:
                    subdiagram = _build_diagram(product, generation+1)
                    diagram = nx.compose(diagram, subdiagram)
                    diagram.add_edge(f'{feynman.typeid}{_multi[feynman.typeid]}', f'{product.typeid}{_multi[product.typeid]}')

            return diagram
        self.diagram = _build_diagram(self)
        return self.diagram

    def draw_diagram(self):
        diagram = self.build_diagram()
        pos = generation_position(diagram)
        color = list(nx.get_node_attributes(diagram,'color').values())
        nx.draw(diagram, pos=pos, with_labels=True, node_color=color)

    @staticmethod
    def _reco_id(feynman):
        if not any(feynman.products): 
            return hash( (feynman.generation, feynman.typeid, feynman.id) )

        hashes = [ Feynman._reco_id(product) for product in feynman.products ]
        return reduce(lambda h1,h2 : hash(str(h1&h2)), hashes)

    def get_reco_id(self, finalstate_ids=None):
        self.build_diagram()
        finalstate = self.get_finalstate()
        if finalstate_ids is None:
            finalstate_ids = np.arange( len(finalstate) )
        for id, state in zip(finalstate_ids, finalstate):
            state.id = id

        return Feynman._reco_id(self)

    def get_finalstate_permutations(self, nobjs=0):
        nfinalstates = len(self.get_finalstate())
        finalstate_ids = np.arange( max(nobjs, nfinalstates) )

        permMap = dict()
        for perm in permutations(finalstate_ids):
            perm = perm[:nfinalstates]
            reco_id = self.get_reco_id(perm)
            if reco_id not in permMap:
                permMap[reco_id] = perm
        
        self.finalstate_permutations = list(permMap.values())
        return self.finalstate_permutations

    def get_multi_reco_id(self, **finalstate_ids):
        self.build_diagram()
        finalstate_types = self.get_finalstate_types()

        for key, states in finalstate_types.items():
            for state, id in zip(states, finalstate_ids[key]):
                state.id = id

        return Feynman._reco_id(self)
    
    def get_multistate_permutations(self, **nfinalstates):

        finalstate_types = self.get_finalstate_types()
        nfinalstate_types = { key:len(finalstate) for key, finalstate in finalstate_types.items() }

        finalstate_ids = { id:np.arange( max(nfinalstates.get(id, nobj), nobj) ) for id, nobj in nfinalstate_types.items() }
        finalstate_permutations = { key:list(permutations(ids)) for key, ids in finalstate_ids.items() }

        permMap = dict()
        for permutation in product(*finalstate_permutations.values()):
            permutation = { key:perm[:nfinalstate] for key, nfinalstate, perm in zip(nfinalstate_types.keys(), nfinalstate_types.values(), permutation)}
            reco_id = self.get_multi_reco_id(**permutation)
            if reco_id not in permMap:
                permMap[reco_id] = permutation

        self.finalstate_permutations = pd.DataFrame(permMap.values()).to_dict(orient='list')
        return self.finalstate_permutations

    def __str__(self, generation=0):
        space = ' '*generation

        string = space+f'{self.typeid}'

        if any(self.products):
            string += ' <\n'

            for product in self.products:
                string += product.__str__(generation+1)

            string += '\n' + space + '>\n'

        return string
    def __repr__(self): return self.typeid