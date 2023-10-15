from pathlib import Path

import yaml

<<<<<<< HEAD
with open("globals.yml", "r") as stream:
=======
with open("rome/globals.yml", "r") as stream:
>>>>>>> bb17df60f7534bc30f80268fde02e6dceedcfc44
    data = yaml.safe_load(stream)

(RESULTS_DIR, DATA_DIR, STATS_DIR, HPARAMS_DIR,) = (
    Path(z)
    for z in [
        data["RESULTS_DIR"],
        data["DATA_DIR"],
        data["STATS_DIR"],
        data["HPARAMS_DIR"],
    ]
)

REMOTE_ROOT_URL = data["REMOTE_ROOT_URL"]
