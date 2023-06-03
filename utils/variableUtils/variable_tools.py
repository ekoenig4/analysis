def tree_variable(f_variable=None, bins=None, xlabel=None):
    def wrap_variable(f_variable):
        f_variable.bins = bins
        f_variable.xlabel = xlabel or f_variable.__name__
        return f_variable
    if f_variable: return wrap_variable(f_variable)
    return wrap_variable

def cache_variable(f_variable=None, bins=None, xlabel=None):

    def wrap_variable(f_variable):
        f_hash = f"_{f_variable.__name__}_{hash(f_variable)}_"
        def cache(tree, **kwargs):
            if cache.hash in tree.fields: return tree[cache.hash]
            tree.extend(**{cache.hash: f_variable(tree, **kwargs)})
            return tree[cache.hash]
        cache.__name__ = f_variable.__name__
        cache.bins = bins
        cache.xlabel = xlabel or f_variable.__name__
        cache.hash = f_hash
        return cache
    if f_variable: return wrap_variable(f_variable)
    return wrap_variable