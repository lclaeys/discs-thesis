# DISCS-THESIS
This repository builds upon DISCS: A Benchmark for Discrete Sampling: [paper](https://openreview.net/pdf?id=oi1MUMk5NF)

## Replica Exchange
This fork adds two new experiment types, RE_Sampling_Experiment and RE_CO_Experiment, which allow the user to incorporate Replica Exchange (RE) in sampling resp. combinatorial optimization experiments. It is compatible with any existing sampler in the package. The extra parameters to configured under config.experiment are num_replicas , min_temperature, max_temperature, adaptive_temps (boolean, whether to use an adaptive temperature schedule) and save_replica_data (boolean, whether to save replica energies and swapping probabilities of experiments). For an example, see <code>experiment_runner.py<\code>.

## Data
The data used in this package could be found [here](https://drive.google.com/drive/u/1/folders/1nEppxuUJj8bsV9Prc946LN_buo30AnDx).
The data contains the following components:
* Graph data of combinatorial optimization problems for different graph types under `/DISCS-DATA/sco/`.
* Model parameters for energy-based models found at `DISCS-DATA/BINARY_EBM` and `DISCS-DATA/RBM_DATA`. Binray ebm is trained on MNIST, Omniglot, and Caltech dataset and binary and categorical RBM are trained on MNIST and Fashion-MNIST dataset. For the language model, you could download the model parameters from [here](https://huggingface.co/bert-base-uncased).
* Text infilling data generated from WT103 and TBC found at `/DISCS-DATA/text_infilling_data/`.
