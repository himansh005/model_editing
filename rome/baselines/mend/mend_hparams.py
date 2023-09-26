from dataclasses import dataclass
import sys

from util.hparams import HyperParams


@dataclass
class MENDHyperParams(HyperParams):
    lr_scale: float
    n_toks: int
    model_name: str
    counterfact: bool
    mini: bool
    zsre: bool
