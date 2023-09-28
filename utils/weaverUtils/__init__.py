from .load import *
# from .onnx import WeaverONNX

from .ort import ONNXRuntimeHelper

class WeaverONNX(ONNXRuntimeHelper):
    def __init__(self, modelpath, onnxdir='export'):

        preprocessing_file = os.path.join(modelpath, onnxdir, 'preprocess.json')
        model_files = [os.path.join(modelpath, onnxdir, 'model.onnx')]
        super().__init__(preprocessing_file, model_files)