# Dr. Zero: Self-Evolving Search Agents without Training Data

[![Paper](https://img.shields.io/badge/Paper-arXiv:2601.07055-b31b1b.svg)](https://arxiv.org/abs/2601.07055)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

This repository contains the code for [**Dr. Zero: Self-Evolving Search Agents without Training Data**](https://arxiv.org/abs/2601.07055). In this work, we introduce Dr. Zero, a framework enabling search agents to effectively self-evolve without any training data. In particular, we design a self-evolution feedback loop where a proposer generates diverse questions to train a solver initialized from the same base model. As the solver evolves, it incentivizes the proposer to produce increasingly difficult yet solvable tasks, thus establishing an automated curriculum to refine both agents. To enhance training efficiency, we also introduce hop-grouped relative policy optimization (HRPO). This method clusters structurally similar questions to construct group-level baselines, effectively minimizing the sampling overhead in evaluating each query's individual difficulty and solvability. Consequently, HRPO significantly reduces the compute requirements for solver training without compromising performance or stability. Extensive experiment results demonstrate that the data-free Dr. Zero matches or surpasses fully supervised search agents, proving that complex reasoning and search capabilities can emerge solely through self-evolution.

## 🚀 Overview

The core idea is to bootstrap a search agent from a base model (e.g., Qwen or Llama) via iterative self-evolution: the agent synthesizes tasks and then learns to solve them in a multi-turn, tool-using environment.

*   **Proposer:** A question generation agent that aims to create hard yet solvable questions and thereby driving the solver improvement.
*   **Solver:** The primary search agent that is trained with synthetic data from the proposer to answer challenging questions using the search tool.
*   **Zero-Data Initialization:** The process starts with zero training data and relies solely on an external search engine (e.g., Wikipedia passage retriever).

<img src=verl/intro.png width=1000>

## 🛠️ Setup & Installation

### 1. Environment

Ensure you have a Python environment with the necessary dependencies (PyTorch, transformers, faiss-gpu, verl==0.5.0, etc.). The rest of the dependencies can be found [here](https://github.com/volcengine/verl/blob/v0.5.0/requirements.txt) and [here](https://github.com/volcengine/verl/blob/v0.5.0/requirements_sglang.txt).

### 2. Search Engine

This framework relies on a local server with a retriever model. Prepare the corpus and build the index before training.

**Download & Index Corpus:**
Execute the following commands to download the Wikipedia English dump and build the faiss index for the retriever (default: `intfloat/e5-base-v2`). More details can be found under the search folder and the [Search-R1 repository](https://github.com/PeterGriffinJin/Search-R1).

```bash
save_path=./corpus
python scripts/download.py --save_path $save_path
cat $save_path/part_* > $save_path/e5_Flat.index
gzip -d $save_path/wiki-18.jsonl.gz
```

## 🏃 Iterative Self-Evolution Workflow

The training process proceeds in iterations (Iter 1, Iter 2, Iter 3...). Each iteration typically consists of three phases:

### Phase 0: Initial Data Preparation

Before the first iteration, prepare the initial prompts for training and the benchmarks for evaluation.

```bash
python process_train.py --local_dir ./data
python process_test.py --local_dir ./data
```

### Iteration 1

**1. Train Proposer:**
Train the proposer agent to generate challenging yet manageable questions for the base solver.

```bash
bash iter1_challenger.sh
```

**2. Synthesize Data:**
Generate training data using the learnt proposer model. Parameters such as model path and sample size can be specified in the script.

```bash
bash iter1_gen_data.sh
```

**3. Train Solver:**
Train the solver agent on the generated synthetic data using GRPO. This optimizes the solver's ability to search and reason over challenging questions.

```bash
bash iter1_solver.sh
```

**4. Convert Solver to HF Format:**
Specify the trained model path and convert the FSDP checkpoint to the HF format. This allows the proposer to load the latest solver for reward estimation in the next training iteration.

```bash
bash convert.sh
```

### Subsequent Iterations (Iter 2, Iter 3...)

Repeat the process using the scripts for the respective iteration. The model checkpoints from the previous iteration are used as the starting point for the next. You may need to modify the iteration number and model paths in the scripts.

*   `iter2_challenger.sh` -> `iter2_gen_data.sh` -> `iter2_solver.sh` -> `convert.sh`
*   `iter3_challenger.sh` -> `iter3_gen_data.sh` -> `iter3_solver.sh` -> `convert.sh`

## Citation
If you find Dr. Zero interesting, please consider citing our paper :)
```
@article{yue2026dr,
  title={Dr. Zero: Self-Evolving Search Agents without Training Data},
  author={Yue, Zhenrui and Upasani, Kartikeya and Yang, Xianjun and Ge, Suyu and Nie, Shaoliang and Mao, Yuning and Liu, Zhe and Wang, Dong},
  journal={arXiv preprint arXiv:2601.07055},
  year={2026}
}
```

## License
The code is released under a non-commercial license. See [LICENSE](LICENSE.md) for more details.

## Acknowledgements
During the implementation we base our code mostly on [Search-R1](https://github.com/PeterGriffinJin/Search-R1) and [VeRL](https://github.com/volcengine/verl). Many thanks to these awesome authors for their great work!
