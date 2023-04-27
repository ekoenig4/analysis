from collections import defaultdict

def f7(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

class CellDependency:
    """
    A dependency graph for notebooks
    """

    _required_graphs = defaultdict(dict)
    _dependency_graphs = defaultdict(lambda:defaultdict(list))

    @classmethod
    def dependency(cls, *required):
        required = [ f.__name__ for f in required ]

        def set_dependency(cell, required=required):
            cls._dependency_graphs[cell.__qualname__.split('.')[0]][cell.__name__] = required
            return cell
        return set_dependency

    @classmethod
    def required(cls, cell):
        cls._required_graphs[cell.__qualname__.split('.')[0]][cell.__name__] = None
        return cell

    @classmethod
    def get_dependency(cls, notebook):
        if not isinstance(notebook, str):
            notebook = notebook.__name__
        return cls._dependency_graphs[notebook]


    @classmethod
    def get_required(cls, notebook):
        if not isinstance(notebook, str):
            notebook = notebook.__name__
        return list(cls._required_graphs[notebook].keys())

    @staticmethod
    def merge(*dependencies):
        """ 
        Merge multiple dependencies into one
        """
        
        dependency_graph = defaultdict(list)

        for dependency in dependencies:
            for cell, required in dependency.dependency_graph.items():
                dependency_graph[cell] += required

        merged_dependency = dependencies[0]
        merged_dependency.dependency_graph = dependency_graph

        return merged_dependency

    def __init__(self, notebook, cells):
        required_graph = CellDependency.get_required(notebook)
        dependency_graph = CellDependency.get_dependency(notebook)

        for required in required_graph:
            idx = cells.index(required)
            for cell in cells[idx+1:]:
                dependency_graph[cell] = f7(dependency_graph[cell] + [required])

        for cell, dependency in dependency_graph.items():
            dependency_graph[cell] = sorted(dependency, key=cells.index)
        self.dependency_graph = dependency_graph

    def build_runlist(self, runlist):
        graph = self.dependency_graph
        def get_dependence(cell):
            cell_dependency = graph.get(cell, dict())
            runlist = []
            for dependence in cell_dependency:
                runlist += get_dependence(dependence)
            runlist += [cell]
            return f7(runlist)
            
        full_runlist = []
        for cell in runlist:
            full_runlist = f7(full_runlist+get_dependence(cell))
        return full_runlist
    
required = CellDependency.required
dependency = CellDependency.dependency