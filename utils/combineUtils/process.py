import random
import shutil
import os
import uproot as ut

def run_popen(cmd, **kwargs):
    """Wrapper function for running shell commands. Throws an error if the command fails.

    Args:
        cmd (str): shell command to be run

    Returns:
        int: process return code
    """

    from subprocess import Popen, PIPE, STDOUT
    p = Popen(cmd, shell=True, stdout=PIPE, stderr=STDOUT, **kwargs)

    log = p.communicate()[0]
    if p.returncode != 0:
        raise RuntimeError('Command failed: %s' % cmd)
    return log

class CombineProcess:
    def __init__(self, path, datacard, model=None, remove_workspace=True):
        self.path = path

        if not os.path.exists(self.path):
            raise ValueError(f'Path {self.path} does not exist')

        self.tmpdir = '%016x' % random.randrange(16**16)
        self.working_dir = os.path.join(self.path, 'workspace', self.tmpdir)

        self.datacard = datacard
        self.model = model
        self.remove_workspace = remove_workspace

    def __enter__(self):
        if not os.path.exists(self.working_dir):
            os.makedirs(self.working_dir)

        with open(os.path.join(self.working_dir, 'datacard.txt'), 'w') as f:
            f.write(self.datacard)

        if self.model:
            self.model.export_to_root(os.path.join(self.working_dir, 'model.root'))

        combine_run = [
            f'cd {self.working_dir}',
            'source /cvmfs/cms.cern.ch/cmsset_default.sh',
            'eval `scramv1 runtime -sh`',
            f'combine -M AsymptoticLimits datacard.txt',
        ]

        with open(f'{self.working_dir}/run.sh', 'w') as f:
            f.write( '\n'.join(combine_run) )

        self.log = run_popen(f'sh {self.working_dir}/run.sh')

        combine_out = f'{self.working_dir}/higgsCombineTest.AsymptoticLimits.mH120.root'

        with ut.open(f'{combine_out}:limit') as tree:
            tree = tree.arrays(['limit','limitErr'])
            limits, limitErr = tree.limit, tree.limitErr

        self.norm_obs_limit, self.norm_obs_limitErr = limits[-1], limitErr[-1]
        self.norm_exp_limit, self.norm_exp_limitErr = limits[:-1], limitErr[:-1]

        if self.model:
            self.obs_limit, self.obs_limitErr = self.norm_obs_limit * self.model.norm, self.norm_obs_limitErr * self.model.norm
            self.exp_limit, self.exp_limitErr = self.norm_exp_limit * self.model.norm, self.norm_exp_limitErr * self.model.norm
        else:
            self.obs_limit, self.obs_limitErr = self.norm_obs_limit, self.norm_obs_limitErr
            self.exp_limit, self.exp_limitErr = self.norm_exp_limit, self.norm_exp_limitErr

        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        if self.remove_workspace:
            shutil.rmtree(self.working_dir, ignore_errors=True)