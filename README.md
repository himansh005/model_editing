<div align="center">
<h1 align="center">
<span style="background-color:white"><img src="https://mcds.cs.cmu.edu/sites/all/themes/mcds2015/images/mcdslogo.png" width="100" /></span>
<br>Self-Correcting Deep Neural Networks</h1>
<h3>Developing robustness checks for editing large language models (LLMs)</h3>
<h5>Himanshu Thakur's Capstone Project (11-632) for the MCDS Program</h3>

<p align="center">
<img src="https://img.shields.io/badge/GNU%20Bash-4EAA25.svg?style=flat-square&logo=GNU-Bash&logoColor=white" alt="GNU%20Bash" />
<img src="https://img.shields.io/badge/tqdm-FFC107.svg?style=flat-square&logo=tqdm&logoColor=black" alt="tqdm" />
<img src="https://img.shields.io/badge/Jupyter-F37626.svg?style=flat-square&logo=Jupyter&logoColor=white" alt="Jupyter" />
<img src="https://img.shields.io/badge/Git-F05032.svg?style=flat-square&logo=Git&logoColor=white" alt="Git" />
<img src="https://img.shields.io/badge/YAML-CB171E.svg?style=flat-square&logo=YAML&logoColor=white" alt="YAML" />
<img src="https://img.shields.io/badge/SciPy-8CAAE6.svg?style=flat-square&logo=SciPy&logoColor=white" alt="SciPy" />
<img src="https://img.shields.io/badge/OpenAI-412991.svg?style=flat-square&logo=OpenAI&logoColor=white" alt="OpenAI" />

<img src="https://img.shields.io/badge/Python-3776AB.svg?style=flat-square&logo=Python&logoColor=white" alt="Python" />
<img src="https://img.shields.io/badge/Rome-27272A.svg?style=flat-square&logo=Rome&logoColor=white" alt="Rome" />
<img src="https://img.shields.io/badge/pandas-150458.svg?style=flat-square&logo=pandas&logoColor=white" alt="pandas" />
<img src="https://img.shields.io/badge/spaCy-09A3D5.svg?style=flat-square&logo=spaCy&logoColor=white" alt="spaCy" />
<img src="https://img.shields.io/badge/NumPy-013243.svg?style=flat-square&logo=NumPy&logoColor=white" alt="NumPy" />
<img src="https://img.shields.io/badge/JSON-000000.svg?style=flat-square&logo=JSON&logoColor=white" alt="JSON" />
</p>
</div>

---

## 📖 Table of Contents
- [📖 Table of Contents](#-table-of-contents)
- [📍 Overview](#-overview)
- [⚙️ Modules](#modules)
- [📂 repository Structure](#-repository-structure)
- [👏 Acknowledgments](#-acknowledgments)

---
A

## 📍 Overview

- This repository builds over the following repositories [ROME](https://github.com/kmeng01/rome) and [LLM-ATTACKS](https://github.com/llm-attacks/llm-attacks). All the project contributions above these are described in the modules section.

---

## 📦 Contributions

- Probing Model Editing
- Semantic Neighbourhood Search
- Adversarially Prompting Edited Models

---

## ⚙️ Contributed Modules 

<details closed><summary>Adversarially_prompting_edited_models</summary>

| File                            | Summary       |
| ---                             | ---           |
| [ppl.py](adversarially_prompting_edited_models/ppl.py)           | Metric to calculated perplexity |

</details>

<details closed><summary>Experiments</summary>

| File                                      | Summary       |
| ---                                       | ---           |
| [parse_results.ipynb](adversarially_prompting_edited_models/experiments/parse_results.ipynb)        | Post-hoc analyses of results from edited model and metrics |
| [run.sh](adversarially_prompting_edited_models/experiments/run.sh)                     | Used for evaluate_neighbours as long running eval job|
| [evaluate_counterfact.py](adversarially_prompting_edited_models/experiments/evaluate_counterfact.py)    | Edits model one at a time, evaluates edited model using the counterfact_evaluator|
| [counterfact_evaluator.py]({adversarially_prompting_edited_models/experiments/counterfact_evaluator.py)   | Metric to evaluate edited model on CounterFact dataset |
| [post_hoc_analyses.py](adversarially_prompting_edited_models/experiments/post_hoc_analyses.py)       | Collection of routines to visualize and analyze results from evaluate_neighbours and evaluate_counterfact |
| [evaluate_neighbours.py](adversarially_prompting_edited_models/experiments/evaluate_neighbours.py)     | Edits model one at a time, evaluates edited model using the neighbourhood_evaluator|
| [neighbourhood_evaluator.py](adversarially_prompting_edited_models/experiments/neighbourhood_evaluator.py) | Semantic Neighbourhood Search Metric |

</details>


<details closed><summary>Adversarial_prefix_search</summary>

| File                                               | Summary       |
| ---                                                | ---           |
| [monuments.py](adversarially_prompting_edited_models/adversarial_prefix_search/monuments.py)                        | A list of monuments in the US and EUROPE, used to edit token space for prefix searches.|
| [single_prefix_search.py]({adversarially_prompting_edited_models/adversarial_prefix_search/single_prefix_search.py)             | Learns a common prefix for finding side-effects of an edited model  |
| [double_prefix_soft_target_search.py]({adversarially_prompting_edited_models/adversarial_prefix_search/double_prefix_soft_target_search.py) | Learns two different prefixs for finding side-effects of an edited model, assuming any soft target |
| [helpers.py]({adversarially_prompting_edited_models/adversarial_prefix_search/helpers.py)                          | Helper functions for prefix search algorithms |
| [double_prefix_search.py]({adversarially_prompting_edited_models/adversarial_prefix_search/double_prefix_search.py)             | Learns two different prefixs for finding side-effects of an edited model, assuming the same target |

</details>


<details closed><summary>Semantic_neighbourhood_analysis</summary>

| File                                               | Summary       |
| ---                                                | ---           |
| [data_paris.json](semantic_neighbourhood_analysis/data_paris.json)                     | Neighbourhood data generation from gpt3.5-turbo, for factual recall anayses |
| [data_ny.json](semantic_neighbourhood_analysis/data_ny.json)                        | Neighbourhood data generation from gpt3.5-turbo, for factual recall anayses  |
| [create_gpt_dataset.py](semantic_neighbourhood_analysis/create_gpt_dataset.py)               | Creates neighbourhood data from gpt3.5-turbo, for factual recall anayses, uses word2vec for neighbours |
| [neighbourhood_search_interactive.py](semantic_neighbourhood_analysis/neighbourhood_search_interactive.py) | Uses word2vec model for getting K neighbours, prompts edited model and measures Realtive Semantic Drift (RSD), allows testing in interactive mode |
| [postprocess_gpt_dataset.py](semantic_neighbourhood_analysis/postprocess_gpt_dataset.py)          | Removes malformed data points generated by create_gpt_dataset.py |
| [neighbourhood_search.ipynb](semantic_neighbourhood_analysis/neighbourhood_search.ipynb)          | Uses word2vec model for getting K neighbours, prompts edited model and measures Realtive Semantic Drift (RSD) |
| [main.py](semantic_neighbourhood_analysis/main.py)                             | Evaluates an edited model on the dataset generated using create_gpt_dataset.py |

</details>

<details closed><summary>Probing_model_editing</summary>

| File                                            | Summary       |
| ---                                             | ---           |
| [probing_hallucinations.ipynb](probing_model_editing/probing_hallucinations.ipynb)     | Notebook to capture logits changes in edited model, also functions to visualize the effects. |
| [compute_logit_change.py](probing_model_editing/compute_logit_change.py)          | Function to capture logits changes in edited model. |
| [factual_storagage_analyses.ipynb](probing_model_editing/factual_storagage_analyses.ipynb) | Notebook to analyse if editing changes fact storage location using causal tracing. |

</details>

### 🔧 Installation

1. Clone the model_editing repository:
```sh
git clone https://github.com/himansh005/model_editing.git
```

2. Change to the project directory:
```sh
cd model_editing
```

3. Install the dependencies: 
```sh
pip install -r requirements.txt
```

---

## 📂 Repository Structure

```sh
└── model_editing/
    ├── adversarially_prompting_edited_models/
    │   ├── adversarial_prefix_search/
    │   │   ├── double_prefix_search.py
    │   │   ├── double_prefix_soft_target_search.py
    │   │   ├── helpers.py
    │   │   ├── monuments.py
    │   │   └── single_prefix_search.py
    │   ├── api_experiments/
    │   │   └── evaluate_api_models.py
    │   ├── demo.ipynb
    │   ├── experiments/
    │   │   ├── configs/
    │   │   ├── eval_scripts/
    │   │   ├── evaluate.py
    │   │   ├── evaluate_individual.py
    │   │   ├── launch_scripts/
    │   │   ├── main.py
    │   │   └── parse_results.ipynb
    │   ├── llm_attacks/
    │   │   ├── base/
    │   │   ├── gcg/
    │   │   └── minimal_gcg/
    │   ├── ppl.py
    │   ├── requirements.txt
    │   └── setup.py
    ├── datasets/
    ├── probing_model_editing/
    │   ├── compute_logit_change.py
    │   ├── factual_storagage_analyses.ipynb
    │   └── probing_hallucinations.ipynb
    ├── requirements.txt
    ├── rome/
    │   ├── CITATION.cff
    │   ├── baselines/
    │   │   ├── efk/
    │   │   ├── ft/
    │   │   ├── kn/
    │   │   └── mend/
    │   ├── dsets/
    │   │   ├── attr_snippets.py
    │   │   ├── counterfact.py
    │   │   ├── knowns.py
    │   │   ├── tfidf_stats.py
    │   │   └── zsre.py
    │   ├── experiments/
    │   │   ├── causal_trace.py
    │   │   ├── counterfact_evaluator.py
    │   │   ├── evaluate.py
    │   │   ├── evaluate_counterfact.py
    │   │   ├── evaluate_neighbours.py
    │   │   ├── neighbourhood_evaluator.py
    │   │   ├── post_hoc_analyses.py
    │   │   ├── py/
    │   │   ├── run.sh
    │   │   ├── summarize.py
    │   │   ├── sweep.py
    │   │   └── traces.py
    │   ├── globals.yml
    │   ├── hparams/
    │   │   ├── FT/
    │   │   ├── KE/
    │   │   ├── KN/
    │   │   ├── MEND/
    │   │   └── ROME/
    │   ├── notebooks/
    │   │   ├── average_causal_effects.ipynb
    │   │   ├── baselines
    │   │   ├── causal_trace.ipynb
    │   │   ├── causal_trace_frozen_mlp_attn.ipynb
    │   │   ├── experiments
    │   │   ├── globals.yml
    │   │   ├── hparams
    │   │   ├── rome
    │   │   ├── rome.ipynb
    │   │   ├── util
    │   │   └── vis/
    │   ├── rome/
    │   │   ├── compute_u.py
    │   │   ├── compute_v.py
    │   │   ├── layer_stats.py
    │   │   ├── repr_tools.py
    │   │   ├── rome_hparams.py
    │   │   ├── rome_main.py
    │   │   └── tok_dataset.py
    │   ├── scripts/
    │   │   ├── causal_trace.sh
    │   │   ├── colab_reqs/
    │   │   ├── collect_layer_stats.sh
    │   │   ├── ipynb_drop_output.py
    │   │   ├── rome.yml
    │   │   ├── setup_clean_ipynb.sh
    │   │   └── setup_conda.sh
    │   └── util/
    │       ├── generate.py
    │       ├── globals.py
    │       ├── hparams.py
    │       ├── logit_lens.py
    │       ├── nethook.py
    │       ├── perplexity.py
    │       └── runningstats.py
    ├── semantic_neighbourhood_analysis/
    │   ├── create_gpt_dataset.py
    │   ├── data_ny.json
    │   ├── data_paris.json
    │   ├── main.py
    │   ├── neighbourhood_search.ipynb
    │   ├── neighbourhood_search_interactive.py
    │   └── postprocess_gpt_dataset.py
    └── setup.sh

```
---


## 👏 Acknowledgments

- Pratyush Maini and Prof. Zachary Lipton, for compute resources and mentorship.

[**Return**](#Top)

