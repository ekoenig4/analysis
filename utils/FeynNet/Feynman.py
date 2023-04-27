import networkx as nx
import itertools as it
from collections import defaultdict
from functools import reduce 
import numpy as np
import numba
from tqdm import tqdm
from . import parallel_tools as parallel

def generation_position(graph):
    posMap=nx.get_node_attributes(graph,'pos')
    gen_pos = defaultdict(list)
    for gen, y in posMap.values():
        gen_pos[gen].append(y)
    gen_shifts = { gen: 1-np.mean(ys)  for gen, ys in gen_pos.items() }
    posMap = { label : (x, y+gen_shifts[x]) for label, (x,y) in posMap.items() }
    return posMap
        
def get_permutation_hash(permutation):
    permutation_hash = np.concatenate( list(permutation.values()), axis=1)
    permutation_hash = np.apply_along_axis( lambda *args : hash(str(args)), 1, permutation_hash)
    return permutation_hash

def get_permutation_hashMap(permutation):
    permutation_hash = get_permutation_hash(permutation)
    return np.vectorize({ int(k):v for v, k in enumerate( permutation_hash ) }.get)

def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n - 1)

def get_invariant_permutations(feynman, permiter, nfinalstate_types, n_jobs=1):
    if n_jobs > 1: return parallel.get_invariant_permutations(feynman, permiter, nfinalstate_types, n_jobs=n_jobs)

    invariant_reco_ids = []
    # invariant_reco_ids = defaultdict(lambda:0)
    finalstate_permutations = defaultdict(list)

    for permutations in permiter:
        permutations = {
            key: permutation[:nfinalstate]
            for key, nfinalstate, permutation in zip(nfinalstate_types.keys(), nfinalstate_types.values(), permutations)
        }
        reco_id = feynman.get_reco_id(**permutations)
        # invariant_reco_ids[reco_id] += 1
        if reco_id not in invariant_reco_ids:
        # if invariant_reco_ids[reco_id] == 1:
            invariant_reco_ids.append(reco_id)
            for typeid, permutation in permutations.items():
                finalstate_permutations[typeid].append(permutation)

    finalstate_permutations = { key: np.array(permutations) for key, permutations in finalstate_permutations.items() }
    return finalstate_permutations

class Feynman:
    def __init__(self, typeid : str):
        self.id = 0
        self.typeid = typeid
        self.products = []

        self._permutation_cache_ = dict()
    def decays(self, *products):
        self.products = [ product if isinstance(product, Feynman) else Feynman(product) for product in products ]
        for product in self.products:
            product.mother = self
        return self

    def build_diagram(self):
        """Build a networkx DiGraph in the direction of decay products

        Returns:
            Feynman: the same object with built diagram
        """
        if getattr(self, 'diagram', None): return self

        _multi = defaultdict(lambda:0)
        _colors = dict()
        def _build_diagram(feynman, generation=0, pos=(0,0)):
            if not hasattr(feynman, 'generation'):
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
        return self

    def draw_diagram(self):
        """Draws the networkx DiGraph
        """
        self.build_diagram()
        diagram = self.diagram
        pos = generation_position(diagram)
        color = list(nx.get_node_attributes(diagram,'color').values())
        nx.draw(diagram, pos=pos, with_labels=True, node_color=color)

    def get_product_types(self):
        """Get a dictionary for each unique particle type that this particle produces
        Each value of the dictionary will contain a list of all the particles that match that type

        Returns:
            dict: dictionary of unique product types for this particle
        """
        self.build_diagram()
        if getattr(self, 'product_types', None): return self.product_types 

        product_types = defaultdict(list)
        for product in self.products:
            product_types[product.typeid].append(product)
        
        self.product_type = product_types
        return self.product_type

    def get_finalstate(self):
        """Get a list of final state particles

        Returns:
            list: final state particles
        """
        self.build_diagram()
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
        """Get a dictionary for each unique particle type in this particles final state

        Returns:
            dict: dictionary of unique finalstate types for this particle
        """
        self.build_diagram()
        if getattr(self, 'finalstate_types', None): return self.finalstate_types

        finalstate = self.get_finalstate()
        self.finalstate_type = defaultdict(list)
        for state in finalstate:
            self.finalstate_type[state.typeid].append(state)
        return self.finalstate_type

    def get_internalstate(self):
        """Get a list of all internalstate particles included itself

        Returns:
            list: internalstate particles
        """
        self.build_diagram()
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
        """Get a dictionary for each unique particle type in thie particles internal state

        Returns:
            dict: dictionary of unique internal state types for this particle
        """
        self.build_diagram()
        if getattr(self, 'internalstate_types', None): return self.internalstate_types

        internalstate = self.get_internalstate()
        self.internalstate_type = defaultdict(list)
        for state in internalstate:
            self.internalstate_type[state.typeid].append(state)
        return self.internalstate_type
        
    def get_generation_types(self):
        """Get a dictionary with a list of particles in each geneneration

        Returns:
            dict: dictionary of particles ordered by generation
        """
        self.build_diagram()
        if getattr(self, 'generation_types', None): return self.generation_types

        generation_types = defaultdict(lambda:defaultdict(list))

        def _generation_types(feynman):

            for product in feynman.products:
                _generation_types(product)

            generation_types[feynman.generation][feynman.typeid].append(feynman)

        _generation_types(self)
        self.generation_types = generation_types
        return self.generation_types        

    @staticmethod
    # @numba.jit
    def _reco_id(feynman):
        """Internal method for Feynman.get_reco_id, a depth first search into the tree

        Args:
            feynman (Feynman): particle defined using Feynman class

        Returns:
            int: hash unique to the ids set for the final state particles
        """
        if not any(feynman.products): 
            return hash( str((feynman.generation, feynman.typeid, feynman.id)) )

        hashes = sorted([ Feynman._reco_id(product) for product in feynman.products ])

        feynman.product_hash = hash(str(hashes))
        feynman.hash = hash( str((feynman.generation, feynman.typeid, feynman.product_hash)) )
        return feynman.hash

    def get_reco_id(self, **finalstate_ids):
        """Calculate a ID for a particular reconstruction

        Returns:
            int: hash for this particular reconstruction
        """
        self.build_diagram()
        finalstate_types = self.get_finalstate_types()

        for key, states in finalstate_types.items():
            for state, id in zip(states, finalstate_ids[key]):
                state.id = id

        return Feynman._reco_id(self)
    
    def get_finalstate_permutations(self, n_jobs=-1, **nfinalstates):
        """Calculate all the permutations of finalstate objects

        Returns:
            dict: Dictionary of permutations for each finalstate type
        """
        self.build_diagram()
        nfinalstates_key = frozenset(nfinalstates.items())
        if nfinalstates_key in self._permutation_cache_: 
            return self._permutation_cache_[nfinalstates_key]

        finalstate_types = self.get_finalstate_types()
        nfinalstate_types = { key:len(finalstate) for key, finalstate in finalstate_types.items() }

        finalstate_ids = { id:np.arange( max(nfinalstates.get(id, nobj), nobj) ) for id, nobj in nfinalstate_types.items() }
        total_perms = np.prod([ factorial(len(ids)) for ids in finalstate_ids.values() ])
        # finalstate_type_permutations = { key:list(it.permutations(ids)) for key, ids in finalstate_ids.items() }
        finalstate_type_permutations = { key:it.permutations(ids) for key, ids in finalstate_ids.items() }

        permiter = it.product(*finalstate_type_permutations.values())

        n_jobs = max(1, int(np.log10(total_perms))) if n_jobs == -1 else n_jobs
        print(n_jobs)
        finalstate_permutations = get_invariant_permutations(self, permiter, nfinalstate_types, n_jobs=n_jobs)

        self._permutation_cache_[nfinalstates_key] = finalstate_permutations
        return finalstate_permutations

    def get_product_finalstate_assignment(self):
        """Get a list of finalstate particles from all product particles

        Returns:
            list: list of finalstate particles for each product particle
        """
        self.build_diagram()
        nfinalstate_types = defaultdict(lambda:-1)

        assignment = []
        for product in self.products:
            product_assignment = defaultdict(list)
            for finalstate in product.get_finalstate():
                nfinalstate_types[finalstate.typeid] += 1
                product_assignment[finalstate.typeid].append( nfinalstate_types[finalstate.typeid] )
            assignment.append( dict(product_assignment) )

        return assignment

    def get_product_permutations(self, **nfinalstates):
        """Calculate all the permutations of product particles

        Returns:
            dict: Dictionary of permutations for each product type
        """
        self.build_diagram()
        finalstate_types = self.get_finalstate_types()    
        products = self.products
        product_types = self.get_product_types()

        # if products are all finalstate objects return all possible permutations
        if all(product_type in finalstate_types for product_type in product_types):
            return self.get_finalstate_permutations(**nfinalstates)

        # get product permutations for all product type diagrams
        product_permutations = { typeid:products[0].get_finalstate_permutations(**nfinalstates) for typeid, products in product_types.items() }
        product_hash = { typeid: get_permutation_hashMap(permutation) for typeid, permutation in product_permutations.items() }

        # get the finalstate permutations for this diagram
        permutations = self.get_finalstate_permutations(**nfinalstates)
        assignments = self.get_product_finalstate_assignment()

        permutation_assignments = [
            { typeid : permutations[typeid][:,assign] for typeid, assign in assignment.items() }
            for assignment in assignments
        ]

        product_permutations = defaultdict(list)
        for product, permutation_assignment in zip(products, permutation_assignments):
            product_permutations[product.typeid].append( product_hash[product.typeid](get_permutation_hash(permutation_assignment)) )
        product_permutations = { typeid:np.stack(assignments, axis=1) for typeid, assignments in product_permutations.items() }

        return product_permutations

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