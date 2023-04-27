import os, yaml

class WeaverModel:
    def __init__(self, path=None, storage=None, basepath=None, desc=None):
        if not os.path.exists(path):
            storage = os.path.join(storage, path)
            path = os.path.join(basepath, path)

        self.path = path
        self.storage = storage
        self.desc = desc

    @property
    def cfg(self):
        cfg = getattr(self, '_cfg', None)
        if cfg is None:
            with open(f'{self.path}/lightning_logs/tb/hparams.yaml', 'r') as f:
                self._cfg = yaml.safe_load(f)
        return self._cfg

        

    def __repr__(self):
        return yaml.dump(self.cfg)

    def __str__(self):
        if self.desc is None:
            return repr(self)
        return self.desc