import yaml, os

import git
GIT_WD = git.Repo('.', search_parent_directories=True).working_tree_dir
__config__ = f"{GIT_WD}/.config.yaml"

if os.path.exists(__config__):
    with open(__config__, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
else:
    config = {}

locals().update(config)
