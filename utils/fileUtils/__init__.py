# from .eos import *
from . import fs_tools as fs

fs.local = fs.mount('')
fs.default = fs.remote('root://cmseos.fnal.gov/', '/eos/uscms/')

fs.repo = fs.repository(
    fs.local,
    fs.default,
)