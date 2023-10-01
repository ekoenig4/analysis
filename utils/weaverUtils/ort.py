import numpy as np
import json
import awkward as ak

def _pad(a, min_length, max_length, value=0, dtype='float32'):
    return ak.fill_none( ak.pad_none(a, min_length, axis=-1, clip=True), value )

    if len(a) > max_length:
        return a[...,:max_length].astype(dtype)
    elif len(a) < min_length:
        x = (np.ones(min_length) * value).astype(dtype)
        x[...,:len(a)] = a.astype(dtype)
        return x
    else:
        return a.astype(dtype)
    
class Preprocessor:

    def __init__(self, preprocess_file, debug_mode=False):
        with open(preprocess_file) as fp:
            self.prep_params = json.load(fp)
        self.debug = debug_mode

    def preprocess(self, inputs):
        data = {}
        for group_name in self.prep_params['input_names']:
            data[group_name] = []
            info = self.prep_params[group_name]
            for var in info['var_names']:
                a = self.preprocess_variable(inputs[var], **info, **info['var_infos'], **info['var_infos'][var])
                if self.debug:
                    print(var, inputs[var], a)
                data[group_name].append( a )

            axis = info.get('jet_dim', 1)
            data[group_name] = np.nan_to_num(np.stack(data[group_name], axis=axis))

            shape = info.get('shape', None)
            if shape is not None:
                data[group_name] = data[group_name].reshape(*shape)
                
        return data

    def preprocess_variable(self, a, 
                            median=None, norm_factor=None, 
                            lower_bound=None, upper_bound=None, 
                            dtype='float32', 
                            replace_inf_value=0, pad=0, 
                            var_length=None, min_length=None, max_length=None, 
                            **info):
        try:
            a = _pad(a, var_length, var_length, dtype=dtype)
        except KeyError:
            a = _pad(a, min_length, max_length, dtype=dtype)
            
        a = np.array(a, dtype=dtype)

        if median is not None:
            a = (a - median)

        if norm_factor is not None:
            a = a * norm_factor

        if lower_bound is not None and upper_bound is not None:
            a = np.clip(a, lower_bound, upper_bound)

        # a = np.expand_dims(a, axis=0)

        return a.astype(dtype)


class ONNXRuntimeHelper:

    def __init__(self, preprocess_file, model_files):
        import onnxruntime
        
        self.preprocessor = Preprocessor(preprocess_file)
        options = onnxruntime.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 1
        options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sessions = [onnxruntime.InferenceSession(model_path, sess_options=options,
                                                      providers=['CPUExecutionProvider']) for model_path in model_files]
        self.k_fold = len(self.sessions)
        self.output_names = [ n for n in self.preprocessor.prep_params['output_names']]
        # print('Loaded ONNX models:\n  %s\npreprocess file:\n  %s' % ('\n  '.join(model_files), str(preprocess_file)))

    def predict(self, inputs, model_idx=None, batch_size=5000):
        data = self.preprocessor.preprocess(inputs)
        if batch_size > len(data): batch_size = None

        if batch_size is None:
            return self.predict_batch(data, model_idx)

        import awkward as ak
        from collections import defaultdict
        
        outputs = defaultdict(list)
        n_batches = max(1, len(data) // batch_size)
        for batch in np.array_split(np.arange(len(data)), n_batches):
            batch = data[batch[0]:batch[-1]]

            for key, array in self.predict_batch(batch, model_idx).items():
                outputs[key].append(array)

        outputs = {k: ak.concatenate(v, axis=0) for k, v in outputs.items()}
        return outputs

    def predict_batch(self, data, model_idx=None):
        if model_idx is not None:
            outputs = self.sessions[model_idx].run([], data)
            outputs = [ out[0] for out in outputs ]
        else:
            outputs = [ sess.run([], data) for sess in self.sessions ]
            outputs = [ np.stack(out, axis=0).mean(axis=0) for out in zip(*outputs) ]
        outputs = {n: v for n, v in zip(self.output_names, outputs)}
        return outputs
