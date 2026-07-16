# Evaluating Communication Topologies in Decentralized Federated Learning

This repository contains the code and results for an undergraduate thesis studying how communication topology affects decentralized federated learning under IID and non-IID data.

## Method

Ten Fashion-MNIST clients train local CNN models and exchange parameters only with graph neighbours. Five topologies are evaluated:

- line
- ring
- star
- hybrid
- fully connected mesh

Neighbour aggregation uses a symmetric doubly-stochastic mixing matrix with Metropolis weights [3]. The training procedure follows the mixing-matrix idea of decentralized parallel stochastic gradient descent, although this implementation performs one local Adam epoch before each communication step and is therefore D-PSGD-inspired rather than an exact reproduction [2].

Client heterogeneity is controlled using a Dirichlet concentration parameter $\alpha$ [4]:

- `100.0`: approximately IID
- `0.5`: moderately non-IID
- `0.1`: highly non-IID

Performance is measured using mean shared-test accuracy, minimum client-model accuracy, convergence over communication rounds, and topology spectral gap. Spectral gap describes how quickly neighbour averaging removes disagreement between client models [2].

A centralized FedAvg implementation is included as a reference baseline [1].

## Repository Structure

- `src/config.py` — experiment settings
- `src/model.py` — Fashion-MNIST CNN
- `src/data.py` — IID and Dirichlet client partitioning
- `src/client.py` — local training
- `src/topology.py` — graph construction, Metropolis weights, and spectral gap
- `src/fedavg_baseline.py` — centralized FedAvg baseline
- `src/train_decentralized.py` — decentralized training
- `src/run_experiments.py` — full experiment sweep
- `results/decentralized_results.csv` — per-round results

## Running

```bash
pip install -r requirements.txt
cd src
python run_experiments.py
```

## References

[1] McMahan et al., “Communication-Efficient Learning of Deep Networks from Decentralized Data,” AISTATS, 2017.

[2] Lian et al., “Can Decentralized Algorithms Outperform Centralized Algorithms? A Case Study for Decentralized Parallel Stochastic Gradient Descent,” NeurIPS, 2017.

[3] Xiao, Boyd, and Lall, “Distributed Average Consensus with Time-Varying Metropolis Weights,” 2006.

[4] Hsu, Qi, and Brown, “Measuring the Effects of Non-Identical Data Distribution for Federated Visual Classification,” 2019.

[5] Yuan et al., “Decentralized Federated Learning: A Survey and Perspective,” IEEE Internet of Things Journal, 2024. The hybrid topology is adapted from the topology examples discussed in this survey.

[6] Xiao, Rasul, and Vollgraf, “Fashion-MNIST: A Novel Image Dataset for Benchmarking Machine Learning Algorithms,” 2017.