from src import *

class TestNotebook(Notebook):
    @staticmethod
    def add_parser(parser):
        parser.add_argument('--var', default='X_m')
        parser.add_argument('--bins', nargs='+', type=float, default=(250,2000,40))
        parser.add_argument('--scale', default='scale')
        return parser

    def __init__(self, var='X_m', bins=(250,2000,40), scale='scale'):
        super().__init__(var=var,bins=bins,scale=scale) 
    
    def build_sf_function(self, trees):
        self.histo = Histo.from_array(getattr(trees, self.var).cat , bins=tuple(self.bins), weights=getattr(trees,self.scale).cat, efficiency=True )

        self.sf_graph = Ratio(Ratio(self.histo, self.histo), self.histo)

        def sf_func(t):
            array = getattr(t, self.var) 
            weights = getattr(t, self.scale)

            sf = self.sf_graph.evaluate(array)
            norm = np.sum(weights)/np.sum(sf*weights)
            return norm * sf * weights

        self.sf_func = sf_func

    def apply_sf_function(self, trees):
        flat_scale = f"{self.scale}_flat_{self.var}"
        trees.apply(lambda t : t.extend(
            **{
                flat_scale:self.sf_func(t)
            }
        ))

        org_scale = getattr(trees, self.scale).cat
        sf_scale = getattr(trees, flat_scale).cat

        print(
            f"Sum of weights ({self.scale}): ({ak.sum( org_scale )})\n"
            f"Sum of weights ({flat_scale}): ({ak.sum( sf_scale )})"
        )

    def write(self, trees):
        for tree in trees:
            tree.write(
                f"flat_{self.var}_{{base}}"
            )

if __name__ == "__main__":
    notebook = TestNotebook.from_parser()
    print(notebook)
    print(notebook.namespace)

