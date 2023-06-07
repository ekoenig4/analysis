import os
import subprocess
import shutil
from glob import glob as local_glob
import re

def ls(path, url, with_path=False):
    cmd = ['eos', url, 'ls', path]
    stdout = subprocess.run(
        [' '.join(cmd)], shell=True, capture_output=True).stdout.decode("utf-8")
    dirlist = stdout.strip().split('\n')

    if with_path:
        path = os.path.dirname(path)
        return [f'{path}/{d}' for d in dirlist]
    return dirlist


def exists(path, url):
    cmd = ['eos', url, 'ls', path]
    stdout = subprocess.run(
        [' '.join(cmd)], shell=True, capture_output=True).stdout.decode("utf-8")
    stdout.strip()
    return any(stdout)