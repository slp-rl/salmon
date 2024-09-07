import os
import random
import numpy as np
import re
from pathlib import Path



def seed_everything(seed: int, deterministic=False):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)



def regex_rglob(path: Path, pattern: str):
    matching_files = []
    for _path in [p for p in path.rglob('*.*')]:
        if re.match(pattern, _path.name):
            matching_files.append(_path)

    return matching_files


