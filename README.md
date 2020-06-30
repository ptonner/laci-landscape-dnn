# Deep neural network modeling of the lacI landscape

## Setup

- Initialize a virtual environment or use global python
- Install `phlanders`:

    ```
    cd phlanders 
    pip install -e .
    ```
    
## Structure
- `data/example`: contains example test and training data similar to that used to train the complete model
- `experiments/example`: configuration files describing the configuration of DNN, dataset, and optimizer used in training each model
  - configurations to run cross-validation and final models under `cv` and `final`, respectively
  - `rnn`: the base recurrent neural network model
  - `bayes-rnn`: approximate Bayesian rnn trained via stochastic variational inference
- `phlanders`: utility library for building and training DNNs on genotype-phenotype landscapes
- `src`: additional code necessary to train and run models
    
## Training models
- models are trained with the command:
  ```
  phlanders --import src.models train --output=output path/to/cfg.json
  ```
- downstream analysis of predictions and accuracy metrics are computed with:
  ```
  phlanders --import src.models predictions --true useBest path/to/cfg.json --results path/to/output/split-0.pt
  ```
- generally these processes can be run in successing for all provided datasets/models with the command:
  ```
  make experiments
  ```
