from src import *

class InitNotebook(Notebook):
    
    @staticmethod
    def add_parser(parser):
        parser.add_argument('--input', nargs='+', default=['../data/Run2017-09Aug2019_UL2017_rsb-v1.root'])
        return parser
    
    # @required
    def init(self):
        print('init')
        self.trees = list(range(10))

class AnotherNotebook(Notebook):
    @staticmethod
    def add_parser(parser):
        parser.add_argument('--input2', nargs='+', default=['../data/Run2017-09Aug2019_UL2017_rsb-v1.root'])
        return parser

    def init2(self):
        print('another init')

class TestNotebook(InitNotebook, AnotherNotebook):

    @staticmethod
    def add_parser(parser):
        parser.add_argument('--var', default='X_m')
        parser.add_argument('--bins', nargs='+', type=float, default=(250,2000,40))
        parser.add_argument('--scale', default='scale')
        return parser

    def init(self):
        print('new init')

    def build_sf_function(self, trees):
        print('build_sf_function', trees)


    def apply_sf_function(self, trees):
        print('apply_sf_function', trees)

    @dependency(apply_sf_function)
    def write(self, trees):
        print('write', trees)

if __name__ == "__main__":
    notebook = TestNotebook.from_parser()
    print(notebook)
    print(notebook.namespace)

    notebook.run()
