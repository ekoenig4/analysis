from .load import *
# from .onnx import WeaverONNX

from .ort import ONNXRuntimeHelper

class WeaverONNX(ONNXRuntimeHelper):
    def __init__(self, modelpath, onnxdir='export'):

        preprocessing_file = os.path.join(modelpath, onnxdir, 'preprocess.json')
        model_files = [os.path.join(modelpath, onnxdir, 'model.onnx')]
        super().__init__(preprocessing_file, model_files)

        self.metadata_file = os.path.join(modelpath, onnxdir, 'metadata.json')

    @property
    def metadata(self):
        if hasattr(self, '_metadata'):
            return self._metadata

        import json
        with open(self.metadata_file) as fp:
            self._metadata = json.load(fp)

        return self._metadata