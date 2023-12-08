import json
import os
import shutil
from pathlib import Path
from time import time
from typing import Tuple, Union
from counter_eval import CounterFactEvaluator
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
import gc
import logging
from glob import glob
import logging
logging.disable(logging.INFO)
import time
logging.getLogger("transformers").setLevel(logging.WARN)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import nltk
nltk.download('punkt')
from baselines.efk import EFKHyperParams, EfkRewriteExecutor
from baselines.ft import FTHyperParams, apply_ft_to_model
from baselines.kn import KNHyperParams, apply_kn_to_model
from baselines.mend import MENDHyperParams, MendRewriteExecutor
from dsets import (
    AttributeSnippets,
    CounterFactDataset,
    MENDQADataset,
    get_tfidf_vectorizer,
)
from experiments.py.eval_utils_counterfact import compute_rewrite_quality_counterfact
from experiments.py.eval_utils_zsre import compute_rewrite_quality_zsre
from rome import ROMEHyperParams, apply_rome_to_model
from util import nethook
from util.globals import *
import flair
from neighbourhood_evaluator import NeighbourhoodEvaluator
flair.device = torch.device('cuda:2')
import random

ALG_DICT = {
    "ROME": (ROMEHyperParams, apply_rome_to_model),
    "FT": (FTHyperParams, apply_ft_to_model),
    "KN": (KNHyperParams, apply_kn_to_model),
    "MEND": (MENDHyperParams, MendRewriteExecutor().apply_to_model),
    "KE": (EFKHyperParams, EfkRewriteExecutor().apply_to_model),
}

DS_DICT = {
    "cf": (CounterFactDataset, compute_rewrite_quality_counterfact),
    "zsre": (MENDQADataset, compute_rewrite_quality_zsre),
}


def main(
    alg_name: str,
    model_name: Union[str, Tuple],
    hparams_fname: str,
    ds_name: str,
    dataset_size_limit: int,
    continue_from_run: str,
    skip_generation_tests: bool,
    conserve_memory: bool,
    dir_name: str,
):
    # Set algorithm-specific variables
    params_class, apply_algo = ALG_DICT[alg_name]
    # Determine run directory
    if continue_from_run is not None:
        run_dir = RESULTS_DIR / dir_name / continue_from_run
        assert (
            run_dir.exists()
        ), f"If continuing from run, {continue_from_run} must exist!"
    else:
        alg_dir = RESULTS_DIR / dir_name
        if alg_dir.exists():
            id_list = [
                int(str(x).split("_")[-1])
                for x in alg_dir.iterdir()
                if str(x).split("_")[-1].isnumeric()
            ]
            run_id = 0 if not id_list else max(id_list) + 1
        else:
            run_id = 0
        run_dir = RESULTS_DIR / dir_name / f"run_{str(run_id).zfill(3)}"
        run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be stored at {run_dir}")
    
    files = glob(os.path.join(run_dir, "*generations.json"))
    case_ids_done = set([int(file.split("/")[-1].split("_")[1]) for file in files])
    # print(case_ids_done)
    print("Cases done:", len(case_ids_done))
    # Get run hyperparameters
    params_path = (
        run_dir / "params.json"
        if continue_from_run is not None
        else HPARAMS_DIR / alg_name / hparams_fname
    )
    hparams = params_class.from_json(params_path)
    if not (run_dir / "params.json").exists():
        shutil.copyfile(params_path, run_dir / "params.json")
    print(f"Executing {alg_name} with parameters {hparams}")

    random.seed(123)
    
    # Instantiate vanilla model
    print("Instantiating model")
    if type(model_name) is str:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model_base = AutoModelForCausalLM.from_pretrained(model_name)
        tok = AutoTokenizer.from_pretrained(model_name)
        model = model.to("cuda:0")
        model_base = model_base.to("cuda:1")
        model_base.eval()
        tok.pad_token = tok.eos_token
    else:
        model, tok = model_name

    
    # Load data
    print("Loading dataset, attribute snippets, tf-idf data")
    snips = AttributeSnippets(DATA_DIR) if not skip_generation_tests else None
    vec = get_tfidf_vectorizer(DATA_DIR) if not skip_generation_tests else None

    ds_class, ds_eval_method = DS_DICT[ds_name]
    ds = ds_class(DATA_DIR, size=dataset_size_limit, tok=tok)
    
    indices = random.sample(range(len(ds)), 10000)

    print("Loading neighbourhood evaluator...")
    ne = NeighbourhoodEvaluator()
    print("Loading complete!")
    
    counter = 0
    for idx, record in enumerate(ds):
        
        case_id = record["case_id"]
        if case_id in case_ids_done:
            continue
        if counter > 1000:
            break
        
        #skip this edit if we cannot evaluate it - in this case not present in w2v vocab.
        try:
            ne.get_head_and_tail_neighbours(record["requested_rewrite"]["subject"])
        except:
            continue

        case_result_path = run_dir / f"case_{case_id}.json"
        generation_path = run_dir / f"case_{case_id}_generations.json"
        
        if not case_result_path.exists():
            # Compute weight changes + record weights that changed
            start = time.time()
            args_conserve_memory = (
                dict(return_orig_weights_device=("cpu" if conserve_memory else "cuda"))
                if conserve_memory
                else dict()
            )

            edited_model, weights_copy = apply_algo(
                model,
                tok,
                [record["requested_rewrite"]],
                hparams,
                copy=False,
                return_orig_weights=True,
                **args_conserve_memory,
            )
            exec_time = time.time() - start
            
            start = time.time()
            metrics = {
                "case_id": case_id,
                "requested_rewrite": record["requested_rewrite"],
                "time": exec_time,
                "post": ds_eval_method(edited_model, tok, record, snips, vec),
            }

            model.eval()
            
            with torch.no_grad():
                for k, v in weights_copy.items():
                    nethook.get_parameter(model, k)[...] = v.to("cuda:0")
            
            record["requested_rewrite"]["id"] = record["case_id"]
            start = time.time()
            nm, generation = ne.run(model, model_base, "cuda:0", "cuda:1", tok, record["requested_rewrite"])
            nm["eval_time"] = time.time() - start
            
            metrics["neighbourhood_metrics"] = nm
            # metrics["pre"] = ds_eval_method(model, tok, record, snips, vec)
            del weights_copy
            
            print("Evaluation took", time.time() - start)

            # # Dump metrics in .json
            with open(case_result_path, "w") as f:
                json.dump(metrics, f, indent=1)
            
            with open(generation_path, "w") as f:
                json.dump(generation, f, indent=1)

            counter += 1

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--alg_name",
        choices=["ROME", "FT", "KN", "MEND", "KE"],
        default="ROME",
        help="Editing algorithm to use. Results are saved in results/<alg_name>/<run_id>, "
        "where a new run_id is generated on each run. "
        "If continuing from previous run, specify the run_id in --continue_from_run.",
        required=True,
    )
    parser.add_argument(
        "--model_name",
        choices=["gpt2-medium", "gpt2-large", "gpt2-xl", "EleutherAI/gpt-j-6B"],
        default="gpt2-xl",
        help="Model to edit.",
        required=True,
    )
    parser.add_argument(
        "--hparams_fname",
        type=str,
        default="gpt2-xl.json",
        help="Name of hyperparameters file, located in the hparams/<alg_name> folder.",
        required=True,
    )
    parser.add_argument(
        "--ds_name",
        choices=["cf", "zsre"],
        default="cf",
        help="Dataset to perform evaluations on. Either CounterFact (cf) or zsRE (zsre).",
    )
    parser.add_argument(
        "--continue_from_run",
        type=str,
        default=None,
        help="If continuing from previous run, set to run_id. Otherwise, leave as None.",
    )
    parser.add_argument(
        "--dataset_size_limit",
        type=int,
        default=10000,
        help="Truncate CounterFact to first n records.",
    )
    parser.add_argument(
        "--skip_generation_tests",
        dest="skip_generation_tests",
        action="store_true",
        help="Only run fast probability-based tests without slow generation tests. "
        "Useful for quick debugging and hyperparameter sweeps.",
    )
    parser.add_argument(
        "--conserve_memory",
        dest="conserve_memory",
        action="store_true",
        help="Reduce memory usage during evaluation at the cost of a minor slowdown. "
        "Backs up model weights on CPU instead of GPU.",
    )
    parser.set_defaults(skip_generation_tests=False, conserve_memory=False)
    args = parser.parse_args()

    main(
        args.alg_name,
        args.model_name,
        args.hparams_fname,
        args.ds_name,
        args.dataset_size_limit,
        args.continue_from_run,
        args.skip_generation_tests,
        args.conserve_memory,
        dir_name=args.alg_name,
    )
    
    
    
    '''
    In some edits, it does not matter if we are correcting the fact or not, it always degrades, seems to be related to
    the subject or relation. Also, disambiguating subject helps.
    '''
    
    '''
    1. Fact correction vs corruptioon
    2. Detect bad edits
    3. Rewriter bad edits
    4. Evaluate for side effects
    5. Use ADV search for re-write
    
    If we reduce side effects, neighbourhood and paraphra score decrease.
    If we insert a plausible correction, scores seems good. however, sometimes, the correction causes leakage into neigbours.
    Problems:
    
    1. No side-effect edits are not good in paraphras and/or neigbourhood scores.
    2. Ones that are good in scores have side effects.
    3. Ones that have side-effects seems to be invariant to fact correction or corruption, but seems related to based model logits.
    4. How to find potential side effects then?
    5. Seems like causal tracing can be used to find the right edit layers (roughly) to reduce side effects, but its not clear what layers or tokens to select.
    6. Subject disambiguation helps, but reduced edit efficacy.
    '''
    
# London is a twin city of


'''
Edit is first evaluated on a frozen dataset
Then, instances where edit fails - those edits are depth wise analyzed for their neighbour using w2v
To try: self moddel correct neighbours.
'''