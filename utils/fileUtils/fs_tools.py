from . import xrd_tools as xrd
from . import local_tools as loc

import os

class store:
    def __init__(self, url = "root://cmseos.fnal.gov/"):
        self.url = url

    def fullpath(self, path):
        return xrd.join_url(path, self.url)

    def cleanpath(self, path):
        return xrd.cleanpath(path, self.url)

    def ls(self, path, with_path=False):
        return xrd.ls(path, self.url, with_path=with_path)

    def glob(self, path, with_path=False):
        return xrd.glob(path, self.url, with_path=with_path)

    def exists(self, path):
        return xrd.exists(path, self.url)

    def copy_from(self, src, dest):
        return xrd.copy(src, dest, src_url=self.url)

    def copy_to(self, src, dest):
        return xrd.copy(src, dest, dest_url=self.url)

    def move(self, src, dest):
        return xrd.move(src, dest, dest_url=self.url)

    def batch_copy_from(self, srcs, dests, nproc=8):
        return xrd.batch_copy(srcs, dests, src_url=self.url, nproc=nproc)

    def batch_copy_to(self, srcs, dests, nproc=8):
        return xrd.batch_copy(srcs, dests, dest_url=self.url, nproc=nproc)

class mount:
    def __init__(self, mnt='/eos/uscms/'):
        self.mnt = mnt

    def fullpath(self, path):
        return loc.join_mnt(path, self.mnt)
    
    def cleanpath(self, path):
        return loc.cleanpath(path, self.mnt)

    def ls(self, path, with_path=False):
        return loc.ls(path, self.mnt, with_path=with_path)

    def glob(self, path, with_path=False):
        return loc.glob(path, self.mnt, with_path=with_path)

    def exists(self, path):
        return loc.exists(path, self.mnt)

    def copy_from(self, src, dest):
        return loc.copy(src, dest, src_mnt=self.mnt)

    def copy_to(self, src, dest):
        return loc.copy(src, dest, dest_mnt=self.mnt)

    def move(self, src, dest):
        return loc.move(src, dest, dest_mnt=self.mnt)

    def batch_copy_from(self, srcs, dests, nproc=8):
        return loc.batch_copy(srcs, dests, src_mnt=self.mnt, nproc=nproc)

    def batch_copy_to(self, srcs, dests, nproc=8):
        return loc.batch_copy(srcs, dests, dest_mnt=self.mnt, nproc=nproc)
    
class remote:
    def __init__(self, url="root://cmseos.fnal.gov/", mnt='/eos/uscms/'):
        self.store = store(url)
        self.mount = mount(mnt) if os.path.exists(mnt) else self.store

    def fullpath(self, path):
        return self.store.fullpath(path)
    
    def cleanpath(self, path):
        return self.store.cleanpath(path)

    def ls(self, path, with_path=False):
        return self.mount.ls(path, with_path=with_path)

    def glob(self, path, with_path=False):
        return self.mount.glob(path, with_path=with_path)

    def exists(self, path):
        return self.mount.exists(path)

    def copy_from(self, src, dest):
        return self.store.copy_from(src, dest)

    def copy_to(self, src, dest):
        return self.store.copy_to(src, dest)

    def move(self, src, dest):
        return self.store.move(src, dest)

    def batch_copy_from(self, srcs, dests, nproc=8):
        return self.store.batch_copy(srcs, dests, nproc=nproc)

    def batch_copy_to(self, srcs, dests, nproc=8):
        return self.store.batch_copy(srcs, dests, nproc=nproc)
    
class repository:
    def __init__(self, *repos):
        self.repos = repos
    def add(self, *repos):
        self.repos += list(repos)

    def get(self, path):
        for repo in self.repos:
            if repo.exists(path):
                return repo
        return None

    def fullpath(self, path):
        for repo in self.repos:
            if repo.exists(path):
                return repo.fullpath(path)
        return None

    def cleanpath(self, path):
        for repo in self.repos:
            if repo.exists(path):
                return repo.cleanpath(path)
        return None
    
    def ls(self, path, with_path=False):
        for repo in self.repos:
            if repo.exists(path):
                return repo.ls(path, with_path=with_path)
        return None
    
    def glob(self, path, with_path=False):
        for repo in self.repos:
            if repo.exists(path):
                return repo.glob(path, with_path=with_path)
        return None
    
    def exists(self, path):
        for repo in self.repos:
            if repo.exists(path):
                return True
        return False
    
    def copy_from(self, src, dest):
        for repo in self.repos:
            if repo.exists(src):
                return repo.copy_from(src, dest)
        return None
    
    def copy_to(self, src, dest):
        for repo in self.repos:
            if repo.exists(src):
                return repo.copy_to(src, dest)
        return None
    