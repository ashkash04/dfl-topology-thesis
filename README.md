# Evaluating Communication Topologies in Decentralized Federated Learning

This repository contains the code and results for an undergraduate thesis studying how communication topology affects decentralized federated learning under IID-like and non-IID client data.

## Method

Ten Fashion-MNIST clients train local CNN models using one local Adam epoch per communication round. Five decentralized communication topologies are evaluated:

- line
- ring
- star
- hybrid
- fully connected mesh

Neighbour aggregation uses symmetric doubly stochastic mixing matrices with Metropolis weights [3]. The training procedure is inspired by the mixing-matrix formulation of decentralized parallel stochastic gradient descent (D-PSGD), but it is not an exact reproduction of D-PSGD [2].

Client-data heterogeneity is controlled using a Dirichlet concentration parameter \(\alpha\) [4]:

- `100.0` — approximately IID
- `0.5` — moderately non-IID
- `0.1` — highly non-IID

The decentralized experiments measure mean shared-test accuracy, minimum client-model accuracy, convergence over communication rounds, and topology spectral gap. Spectral gap describes how quickly neighbour averaging removes disagreement between client models [2].

A matched, unweighted FedAvg baseline is also evaluated using the same model, Dirichlet partitions, local training procedure, alpha values, random seeds, and number of rounds [1].

## Repository Structure

- `src/config.py` — experiment settings
- `src/model.py` — Fashion-MNIST CNN
- `src/data.py` — IID and Dirichlet client partitioning
- `src/client.py` — local client training
- `src/topology.py` — graph construction, Metropolis weights, and spectral gap
- `src/fedavg_baseline.py` — FedAvg aggregation and evaluation functions
- `src/train_decentralized.py` — decentralized training and neighbour-mixing functions
- `src/run_experiments.py` — full decentralized topology sweep
- `src/run_fedavg.py` — matched FedAvg sweep
- `results/decentralized_results.csv` — per-round decentralized results
- `results/fedavg_results.csv` — per-round FedAvg results

## Running

```bash
pip install -r requirements.txt
cd src

python run_experiments.py
python run_fedavg.py
```

## Result Columns

`decentralized_results.csv`:

```text
topology, alpha, seed, spectral_gap, round, avg_acc, worst_acc
```

`fedavg_results.csv`:

```text
alpha, seed, round, accuracy
```

## References

[1] McMahan et al., “Communication-Efficient Learning of Deep Networks from Decentralized Data,” AISTATS, 2017.

[2] Lian et al., “Can Decentralized Algorithms Outperform Centralized Algorithms? A Case Study for Decentralized Parallel Stochastic Gradient Descent,” NeurIPS, 2017.

[3] Xiao, Boyd, and Lall, “Distributed Average Consensus with Time-Varying Metropolis Weights,” 2006.

[4] Hsu, Qi, and Brown, “Measuring the Effects of Non-Identical Data Distribution for Federated Visual Classification,” 2019.

[5] Yuan et al., “Decentralized Federated Learning: A Survey and Perspective,” *IEEE Internet of Things Journal*, 2024. The hybrid topology is adapted from topology examples discussed in this survey.

[6] Xiao, Rasul, and Vollgraf, “Fashion-MNIST: A Novel Image Dataset for Benchmarking Machine Learning Algorithms,” 2017.