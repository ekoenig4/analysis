import os

class resources:
    environ = os.environ
    slurm_environ = {k: v for k, v in environ.items() if k.startswith('SLURM')}

    ncpus = len(os.sched_getaffinity(0))

    @property
    def ngpus(self):
        import utils.compat.torch as torch
        return torch.cuda.device_count()
    
resources = resources()


