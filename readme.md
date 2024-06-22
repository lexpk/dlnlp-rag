This repository contains the code for our project for the course "Deep Learning for Natural Language Processing" at TU Wien. Our chosen topic is the paper ["Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"](https://arxiv.org/abs/2005.11401).

## Setup

The recommended way to install the requirements is using a conda environment. You can create and activate a new environment with the required packages using the following command:

```bash
conda env create -f environment.yml
conda activate rag
```

You may want to change cuda/pytorch version and the environment name in the `environment.yml` file.

After activating the environment you can install the package with the following command:

```bash
pip install -e .
```

This will install the package in editable mode, so you can modify the code and the changes will be reflected in the installed package.

To download the models form huggingface you need to agree to the [llama license agreement](https://huggingface.co/meta-llama/Meta-Llama-3-8B) and add your [access token](https://huggingface.co/docs/hub/security-tokens) as an environment variable. Simply create a file `.env` containing

```
HF_TOKEN=your_token_goes_here
```

If you want to use the WebSearch Retriever you have to add an additional environment variable with your [Tavily](https://docs.tavily.com/docs/tavily-api/introduction) token

```
WEB_SEARCH_TOKEN=your_token_goes_here
```

## Project Structure

The main entry point of the project is the `experiments.ipynb` notebook.

## Bio Rag experiement

The code for the bio experiment is provided in `bio_experiement.ipynb`.

### Change dataset

To change the dataset adapt:

- `plot_dir` ... this sets the path for the generated plots
- `dataset` ... to load the correct dataset
- `get_query` and `get_solution` lamda functions ... this adapts how the query/question and the solution are extracted from the dataset
