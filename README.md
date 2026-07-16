# Evaluating Communication Topologies in Decentralized Federated Learning under IID and Non-IId Data Assumptions

Undergraduate thesis (Western University). A simulation study of how the
communication topology between clients in a distributed network affects
decentralized federated learning, especially under heterogeneous (non-IID) data.

## Overview

Five communication topologies (line, ring, star, hybrid, mesh) are compared
against a centralized FedAvg baseline. Each topology is expressed as a
doubly-stochastic mixing matrix (Metropolis weights); clients train locally and
average models only with their neighbours, with D-PSGD. Data heterogeneity is
controlled by a Dirichlet concentration parameter $\alpha$. Performance is measured
by average and worst-client accuracy, and related to each topology's spectral gap.

## Structure

- `config.py` - all hyperparameters and experimental factors
- `model.py` - the CNN
- `data.py` - Fashion-MNIST loading; IID and Dirichlet non-IID partitioning
- `client.py` - a single client's local training step
- `topology.py` - topologies as mixing matrixes; spectral-gap computation
- `fedavg_baseline.py` - centralized FedAvg baseline (demo)
- `train_decentralized.py` - decentralized training loop (demo + shared functions)
- `run_experiments.py` - full sweep over topologies * alpha * seeds to the CSV

## Running

```
cd src
python run_experiments.py       # full experiment sweep, writes results to CSV
python fedavg_baseline.py                 # centralized baseline demo
python train_decentralized.py   # single-topology decentralized demo
```
