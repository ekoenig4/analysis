import os, torch

environ = os.environ
slurm_environ = {k: v for k, v in environ.items() if k.startswith('SLURM')}

ncpus = len(os.sched_getaffinity(0))
ngpus = torch.cuda.device_count()