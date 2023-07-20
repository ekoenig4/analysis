import os
import subprocess
import shutil
from glob import glob as local_glob
import multiprocessing as mp
import re

def cleanpath(path, mnt=None):
    if mnt and mnt in path:
        path = path[len(mnt):]

    extra_slash = re.compile(r'//+')
    for extra in set(extra_slash.findall(path)):
        path = path.replace(extra,'/')

    return path

def join_mnt(path, mnt=None):
    if mnt:
        return mnt + path
    return path

def glob(path, mnt=None, with_path=False):
    path = join_mnt(path, mnt)
    dirlist = [ f[len(mnt):] for f in local_glob(path) ]
    
    if with_path:
        path = os.path.dirname(path)
        return [ f'{path}/{os.path.basename(d)}' for d in dirlist]
    
    return dirlist

def ls(path, mnt=None, with_path=False):
    base = path
    path = join_mnt(path, mnt)
    dirlist = [ f for f in os.listdir(path) ]
    
    if with_path:
        return [ f'{base}/{os.path.basename(d)}' for d in dirlist]

def exists(path, mnt=None):
    path = join_mnt(path, mnt)
    return os.path.exists(path)

def copy(src, dest, src_mnt=None, dest_mnt=None):
    src = join_mnt(src, src_mnt)
    dest = join_mnt(dest, dest_mnt)

    return shutil.copy2(src, dest)

def move(src, dest, src_mnt=None, dest_mnt=None):
    src = join_mnt(src, src_mnt)
    dest = join_mnt(dest, dest_mnt)

    status = copy(src, dest)

    if status:
        os.remove(src)
    return src

def batch_copy(srcs, dests, src_mnt=None, dest_mnt=None, nproc=8):
    nsrcs, ndests = len(srcs), len(dests)
    assert nsrcs == ndests, f'Number of srcs={nsrcs} should match number of dests={ndests}'

    # prepare the arguments for multiprocessing
    args = zip(srcs, 
               dests, 
               [src_mnt]*nsrcs, 
               [dest_mnt]*ndests)
    
    with mp.pool.ThreadPool(nproc) as pool:
        pool.starmap(copy, args)