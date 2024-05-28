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

## Project Structure

The main entry point of the project is the `experiments.ipynb` notebook.