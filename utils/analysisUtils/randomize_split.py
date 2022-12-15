from .. import *


class randomize_split(Analysis):
    @staticmethod
    def _add_parser(parser):
        parser.add_argument("--nsplit", default=5, type=int)
        return parser

    def randomize_split(self, trees):
        (trees).apply(
            lambda t : t.extend(
                _random_split= ak.from_numpy( np.random.randint(self.nsplit, size=len(t)) )
            )
        )

    def write_split(self, trees):
        for i in range(self.nsplit):
            split = trees.copy()
            split = split.apply(EventFilter(f'split_{i}', filter=lambda t : t._random_split==i))

            study.quick( 
                split,
                varlist=['jet_pt[:,0]']
            )
            split.write(f'split_{i}_{{base}}', include=['^jet','^X','.*scale$','is_bkg'])
