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
        train_split.write(f'train_{{base}}')

        test_split = trees.copy()
        test_split = test_split.apply(EventFilter(f'test', filter=lambda t : ~t._random_split))

        study.quick( 
            test_split,
            varlist=['jet_pt[:,0]']
        )
        test_split.write(f'test_{{base}}')
