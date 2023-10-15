from dataclasses import dataclass
<<<<<<< HEAD
import sys
=======
>>>>>>> bb17df60f7534bc30f80268fde02e6dceedcfc44

from util.hparams import HyperParams


@dataclass
class MENDHyperParams(HyperParams):
    lr_scale: float
    n_toks: int
    model_name: str
    counterfact: bool
    mini: bool
    zsre: bool
