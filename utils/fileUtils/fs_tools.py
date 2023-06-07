from . import xrd_tools as xrd
from . import local_tools as lcl

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
        return lcl.join_mnt(path, self.mnt)
    
    def cleanpath(self, path):
        return lcl.cleanpath(path, self.mnt)

    def ls(self, path, with_path=False):
        return lcl.ls(path, self.mnt, with_path=with_path)

    def glob(self, path, with_path=False):
        return lcl.glob(path, self.mnt, with_path=with_path)

    def exists(self, path):
        return lcl.exists(path, self.mnt)

    def copy_from(self, src, dest):
        return lcl.copy(src, dest, src_mnt=self.mnt)

    def copy_to(self, src, dest):
        return lcl.copy(src, dest, dest_mnt=self.mnt)

    def move(self, src, dest):
        return lcl.move(src, dest, dest_mnt=self.mnt)

    def batch_copy_from(self, srcs, dests, nproc=8):
        return lcl.batch_copy(srcs, dests, src_mnt=self.mnt, nproc=nproc)

    def batch_copy_to(self, srcs, dests, nproc=8):
        return lcl.batch_copy(srcs, dests, dest_mnt=self.mnt, nproc=nproc)
    
class remote:
    def __init__(self, url="root://cmseos.fnal.gov/", mnt='/eos/uscms/'):
        self.store = store(url)
        self.mount = mount(mnt)

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
        return self.store.copy(src, dest)

    def copy_to(self, src, dest):
        return self.store.copy(src, dest)

    def move(self, src, dest):
        return self.store.move(src, dest)

    def batch_copy_from(self, srcs, dests, nproc=8):
        return self.store.batch_copy(srcs, dests, nproc=nproc)

    def batch_copy_to(self, srcs, dests, nproc=8):
        return self.store.batch_copy(srcs, dests, nproc=nproc)
    