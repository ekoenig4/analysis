# Study framework for NanoHH4b
This module is a framework for producing studies for the NanoHH4b analysis.

# Installing
Before running, make sure to run the install script to install the necessary packages in the CMSSW environment:
```
cmsenv
sh install.sh
```

## User Config
There is a user config file `.config.yaml` that is used to set a global variables used by the framework.


## Main Analysis
The main analysis is run with the script `scripts/nanohh4b/nanohh4b.py`, which is accompanied by a config script `configs/nanohh4b/nanohh4b.yaml`. The analysis script is a collection of methods that are run sequentially. To see all available methods, run:
```
python3 scripts/nanohh4b/nanohh4b.py --dry-run
```

A single method (or list of methods) can be select to run using the --only options:
```
python3 scripts/nanohh4b/nanohh4b.py --only <method1> <method2> ...
```

There is a dependency graph that is defined in the file that will activate all necessary methods to run the specified method. To run all methods, run:
```
python3 scripts/nanohh4b/nanohh4b.py
```

### Changing Configs
By default the config file `configs/nanohh4b/nanohh4b.yaml` is used. To use a different config file, use the `--config` option:
```
python3 scripts/nanohh4b/nanohh4b.py --config <config_file>
```