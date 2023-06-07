import os
import subprocess
import shutil
from glob import glob as local_glob
import multiprocessing as mp
import re

def cleanpath(path, url=None):
    if url and url in path:
        path = path[len(url):]

    extra_slash = re.compile(r'//+')
    for extra in set(extra_slash.findall(path)):
        path = path.replace(extra,'/')
    return path

def join_url(path, url=None):
    if url:
        return url + path
    return path

def ls(path, url, with_path=False):
    cmd = ['xrdfs', url, 'ls', path]
    stdout = subprocess.run(
        [' '.join(cmd)], shell=True, capture_output=True).stdout.decode("utf-8")
    dirlist = stdout.strip().split('\n')

    if with_path:
        path = os.path.dirname(path)
        return [f'{path}/{d}' for d in dirlist]
    return dirlist

def glob(path, url, with_path=False):
    path = cleanpath(path)

    blocks = path.split('/')
    nblocks = len(blocks)
    head = next( (i for i, block in enumerate(blocks) if '*' in block), len(blocks))
    head = '/'.join(blocks[:head])

    cmd = ['xrdfs', url, 'ls', '-R', head]
    stdout = subprocess.run(
        [' '.join(cmd)], shell=True, capture_output=True).stdout.decode("utf-8")
    dirlist = stdout.strip().split('\n')

    pattern = path.replace('*','.*')
    dirlist = [ d for d in dirlist if re.match(pattern, d) and len(d.split('/')) == nblocks ]

    if with_path:
        path = os.path.dirname(path)
        return [ join_url(f'{path}/{os.path.basename(d)}', url) for d in dirlist]
    return dirlist

def exists(path, url):
    dirlist = ls(path, url)
    return any(dirlist)

def copy(src, dest, src_url=None, dest_url=None):
    src = join_url(src, src_url)
    dest= join_url(dest, dest_url)

    cmd = ['xrdcp','-f', src, dest]
    return subprocess.run(
        [' '.join(cmd)], shell=True)

def move(src, dest, src_url=None, dest_url=None):
    status = copy(src, dest, src_url, dest_url)
    if status:
        os.remove(src)

    return status

def batch_copy(srcs, dests, src_url=None, dest_url=None, nproc=8):
    nsrcs, ndests = len(srcs), len(dests)
    assert nsrcs == ndests, f'Number of srcs={nsrcs} should match number of dests={ndests}'

    # prepare the arguments for multiprocessing
    args = zip(srcs, 
               dests, 
               [src_url]*nsrcs, 
               [dest_url]*ndests)
    
    with mp.pool.ThreadPool(nproc) as pool:
        pool.starmap(copy, args)

    

