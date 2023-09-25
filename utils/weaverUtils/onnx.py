from collections import defaultdict
import itertools
import json, os
import numpy as np
import awkward as ak
import multiprocess as mp

variable_map = dict(
    jet_cosphi = lambda t : np.cos(t.jet_phi),
    jet_cos_phi = lambda t : np.cos(t.jet_phi),
    jet_sinphi = lambda t : np.sin(t.jet_phi),
    jet_sin_phi = lambda t : np.sin(t.jet_phi),
    jet_mass = lambda t : t.jet_m,
    jet_m = lambda t : t.jet_mass,
)

def preprocess_variable(array, 
                        length, 
                        median, 
                        norm_factor, 
                        lower_bound, 
                        upper_bound, 
                        pad, 
                        replace_inf_value):
    
    array = norm_factor * (array - median)
    array = ak.fill_none(ak.pad_none(array, length, axis=1, clip=True), pad)
    array = ak.to_numpy(array)
    array = np.clip(array, lower_bound, upper_bound)

    is_inf = np.isinf(array)
    if np.any(is_inf):
        array[is_inf] = replace_inf_value

    return array

class WeaverONNX:
    def add_variable(self, **f_vars):
        self.variable_map.update(f_vars)
    def reset_variables(self):
        self.variable_map = dict(variable_map)

    def __init__(self, modelpath, onnxdir='export'):
        import onnxruntime as ort
    
        """
        Args:
            modelpath (str): Path to the model directory. Should contain the export directory with the model.onnx and preprocess.json files
            variable_map (dict): Dictionary of functions to apply to each variable
        """
        self.modelpath = modelpath
        self.variable_map = dict(variable_map)
        self.onnxdir = onnxdir

    def __call__(self, tree, batch_size=1000):
        inputs = self.get_inputs(tree)

        if batch_size is None or len(tree) < batch_size: return self.predict(inputs)
        from ..utils import get_batch_ranges

        batch_idx = get_batch_ranges(len(tree), batch_size=batch_size)
        results = defaultdict(list)

        nthreads = min( 10, len(batch_idx) - 1)
        with mp.pool.ThreadPool(nthreads) as pool:
            for result in pool.imap(self.thread_predict, zip(itertools.repeat(inputs), batch_idx[:-1], batch_idx[1:])):
                for key, value in zip(self.output_names, result):
                    results[key].append(value)

        result = {k:np.concatenate(v) for k,v in results.items()}
        return result
    
    def thread_predict(self, args):
        inputs, start, stop = args
        batch_input = {k:v[start:stop] for k,v in inputs.items()}
        return self.session.run(None, batch_input)
    
    def predict(self, inputs):
        outputs = self.session.run(None, inputs)
        return dict(zip(self.output_names, outputs))

    def get_inputs(self, tree):
        inputs = {}
        preprocess = self.preprocess
        for input_name in self.input_names:
            input_table = []
            input_info = dict(length=preprocess[input_name]['var_length'])
            for variable in preprocess[input_name]['var_names']:
                f_var = self.variable_map.get(variable, lambda t : t[variable])
                var_info = dict(input_info, **preprocess[input_name]['var_infos'][variable])
                input_table.append(preprocess_variable(f_var(tree), **var_info))
            input_table = np.stack(input_table, axis=1)
            inputs[input_name] = input_table.astype(np.float32)

        return inputs

    @property
    def preprocess(self):
        if getattr(self, '_preprocess', None): return self._preprocess
        with open(os.path.join(self.modelpath, self.onnxdir, 'preprocess.json')) as f:
            self._preprocess = json.load(f)
        return self._preprocess

    @property
    def session(self):
        import onnxruntime as ort
        
        if getattr(self, '_session', None): return self._session
        self._session = ort.InferenceSession(os.path.join(self.modelpath, self.onnxdir, 'model.onnx'))
        return self._session
    
    @property
    def input_names(self):
        if getattr(self, '_input_names', None): return self._input_names
        self._input_names = [i.name for i in self.session.get_inputs()]
        return self._input_names
    
    @property
    def output_names(self):
        if getattr(self, '_output_names', None): return self._output_names
        self._output_names = [i.name for i in self.session.get_outputs()]
        return self._output_names