# from .eos import *

from .fileUtils import FileCollection
from . import fs_tools as fs

fs.local = fs.mount('')
fs.default = fs.remote('root://cmseos.fnal.gov/', '/eos/uscms/')

fs.repo = fs.repository(
    fs.local,
    fs.default,
)

eightb = FileCollection('/store/user/ekoenig/8BAnalysis/NTuples/2018/')