from .. import *
from .. import eightbUtils as eightb


class train_test_split(Analysis):
    @staticmethod
    def _add_parser(parser):
        parser.add_argument("--train", default=0.8, type=float)
        return parser

    def randomize_split(self, trees):
        (trees).apply(
            lambda t : t.extend(
                _random_split= ak.from_numpy( np.random.random(size=len(t)) < self.train )
            )
        )

    def write_split(self, trees):
        train_split = trees.copy()
        train_split = train_split.apply(EventFilter(f'train', filter=lambda t : t._random_split))

        study.quick( 
            train_split,
            varlist=['jet_pt[:,0]']
        )

        def move_to_dir(f, dir):
            f = fc.cleanpath(f)
            f =  f.replace('/output/',f'/{dir}/')
            path = os.path.dirname(f)
            fc.mkdir_eos(path)
            return f

        train_split.write(lambda f : move_to_dir(f, 'train'))

        test_split = trees.copy()
        test_split = test_split.apply(EventFilter(f'test', filter=lambda t : ~t._random_split))

        study.quick( 
            test_split,
            varlist=['jet_pt[:,0]']
        )
        test_split.write(lambda f : move_to_dir(f, 'test'))
