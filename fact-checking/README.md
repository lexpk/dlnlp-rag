# Fact Verification

## Installation

To get started, open a terminal and switch to the `fact-checking` directory. Then, run `conda env create -f environment.yml` to create a new conda environment with all the necessary dependencies. Finally, you have to add the environment to Jupyter by running `python -m ipykernel install --user --name=rag`.

## Running the code

First, you have to execute the code contained in `gen-data.ipynb`to generate the FEVER dataset with claims and fine- and coarse-grained evidence. After executing the notebook, there should be a directory called `fever-fine-coarse` which contains the dataset.

Then, you can run the the code contained in the `experiments.ipynb` notebook to perform the experiments.
