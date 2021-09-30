import tensorflow as tf
from tensorflow import keras
import json
from configparser import ConfigParser
import numpy as np


class Model:
    def __init__(self, modeldir):
        self.modeldir = modeldir
        self.cfg = ConfigParser()
        with open(modeldir+'/model.cfg', 'r') as f_cfg:
            self.cfg.read_file(f_cfg)
        self.scale_min = np.array(self.cfg.get(
            'scaler', 'scale_min').split(','), dtype=float)
        self.scale_max = np.array(self.cfg.get(
            'scaler', 'scale_max').split(','), dtype=float)

        with open(modeldir+'/model.json', 'r') as f_json:
            self.arch = json.load(f_json)
            self.model = keras.models.model_from_json(json.dumps(self.arch))
        self.model.load_weights(modeldir+'/model.h5')
        self.model.compile()

    def __getattr__(self, key): return getattr(self.model, key)

    def __str__(self):
        string = []
        for seq in self.cfg:
            string.append(f"[{seq}]")
            for key, value in self.cfg.items(seq):
                string.append(f"{key}={value}")
            string.append("\n")
        return '\n'.join(string)

    def predict(self, features, *args, **kwargs):
        scaled_features = (features-self.scale_min) / \
            (self.scale_max-self.scale_min)
        return self.model.predict(scaled_features, *args, **kwargs)
