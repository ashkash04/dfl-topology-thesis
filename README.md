# Evaluating Communication Topologies in Decentralized Federated Learning

This repository contains the code, results, figures, and final undergraduate thesis studying how communication topology affects decentralized federated learning under IID-like and non-IID client data.

## Method

Ten Fashion-MNIST clients train local CNN models using one local Adam epoch per communication round. Five decentralized communication topologies are evaluated:

- line
- ring
- star
- hybrid
- fully connected mesh

Neighbour aggregation uses symmetric doubly stochastic mixing matrices with Metropolis weights [13]. The training procedure is inspired by decentralized parallel stochastic gradient descent (D-PSGD), but is more accurately described as decentralized federated averaging because each client completes a full local epoch before each communication step [9], [10].

Client-data heterogeneity is controlled using a Dirichlet concentration parameter (\alpha) [6]:

- `100.0` — approximately IID
- `0.5` — moderately non-IID
- `0.1` — highly non-IID

Each configuration is trained for 50 communication rounds across ten random seeds.

The decentralized experiments measure mean test accuracy, minimum client-model accuracy, convergence over communication rounds, and topology spectral gap.

Two centralized FedAvg baselines are also evaluated using the same model, Dirichlet partitions, local training procedure, alpha values, random seeds, and number of rounds [1]:

- equal-client FedAvg — matched to the fully connected mesh's equal-client averaging rule
- sample-weighted FedAvg — weights client models by local dataset size

## Repository Structure

- `dfl_topology_thesis.pdf` — the full thesis
- `src/config.py` — experiment settings
- `src/model.py` — Fashion-MNIST CNN
- `src/data.py` — IID and Dirichlet client partitioning
- `src/client.py` — local client training
- `src/topology.py` — graph construction, Metropolis weights, and spectral gap
- `src/fedavg_baseline.py` — equal-client and sample-weighted FedAvg aggregation and evaluation
- `src/train_decentralized.py` — decentralized training and neighbour-mixing functions
- `src/run_decentralized.py` — full decentralized topology sweep
- `src/run_fedavg_unweighted.py` — matched equal-client FedAvg sweep
- `src/run_fedavg_weighted.py` — sample-weighted FedAvg sweep
- `src/make_figures.py` — generate figures from completed experiment results
- `results/decentralized_results.csv` — decentralized results
- `results/fedavg_results.csv` — equal-client FedAvg results
- `results/fedavg_weighted_results.csv` — sample-weighted FedAvg results
- `figures/` — generated thesis figures

## Running

```bash
pip install -r requirements.txt
cd src

python run_decentralized.py
python run_fedavg_unweighted.py
python run_fedavg_weighted.py
python make_figures.py
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

`fedavg_weighted_results.csv`:

```text
alpha, seed, round, accuracy
```

## References

[1] H. B. McMahan, E. Moore, D. Ramage, S. Hampson, and B. A. y Arcas, “Communication-Efficient Learning of Deep Networks from Decentralized Data,” *Proceedings of the 20th International Conference on Artificial Intelligence and Statistics*, vol. 54, pp. 1273–1282, 2017.

[2] P. Kairouz, H. B. McMahan, B. Avent, A. Bellet, M. Bennis, A. N. Bhagoji, K. Bonawitz, Z. Charles, G. Cormode, R. Cummings, et al., “Advances and Open Problems in Federated Learning,” *Foundations and Trends in Machine Learning*, vol. 14, no. 1–2, pp. 1–210, 2021.

[3] L. Yuan, Z. Wang, L. Sun, P. S. Yu, and C. G. Brinton, “Decentralized Federated Learning: A Survey and Perspective,” *IEEE Internet of Things Journal*, vol. 11, no. 21, pp. 34617–34638, 2024.

[4] K. Hsieh, A. Phanishayee, O. Mutlu, and P. B. Gibbons, “The Non-IID Data Quagmire of Decentralized Machine Learning,” *Proceedings of the 37th International Conference on Machine Learning*, vol. 119, pp. 4387–4398, 2020.

[5] A. Bellet, A.-M. Kermarrec, and E. Lavoie, “D-Cliques: Compensating for Data Heterogeneity with Topology in Decentralized Federated Learning,” arXiv:2104.07365, 2021.

[6] T.-M. H. Hsu, H. Qi, and M. Brown, “Measuring the Effects of Non-Identical Data Distribution for Federated Visual Classification,” arXiv:1909.06335, 2019.

[7] T. Li, A. K. Sahu, M. Zaheer, M. Sanjabi, A. Talwalkar, and V. Smith, “Federated Optimization in Heterogeneous Networks,” *Proceedings of Machine Learning and Systems*, vol. 2, 2020.

[8] S. Caldas, S. M. K. Duddu, P. Wu, T. Li, J. Konečný, H. B. McMahan, V. Smith, and A. Talwalkar, “LEAF: A Benchmark for Federated Settings,” arXiv:1812.01097, 2019.

[9] X. Lian, C. Zhang, H. Zhang, C.-J. Hsieh, W. Zhang, and J. Liu, “Can Decentralized Algorithms Outperform Centralized Algorithms? A Case Study for Decentralized Parallel Stochastic Gradient Descent,” *Advances in Neural Information Processing Systems*, vol. 30, pp. 5331–5341, 2017.

[10] T. Sun, D. Li, and B. Wang, “Decentralized Federated Averaging,” arXiv:2104.11375, 2021.

[11] I. Hegedűs, G. Danner, and M. Jelasity, “Gossip Learning as a Decentralized Alternative to Federated Learning,” in *Distributed Applications and Interoperable Systems*, Springer, pp. 74–90, 2019.

[12] W.-C. Chung, C.-A. Lo, Y.-H. Lin, Z.-H. Chen, and C.-L. Hung, “Decentralized Federated Learning with Non-IID Data: Challenges, Trends, and Future Opportunities,” *ACM Computing Surveys*, vol. 58, no. 8, pp. 192:1–192:41, 2026.

[13] L. Xiao, S. Boyd, and S. Lall, “Distributed Average Consensus with Time-Varying Metropolis Weights,” unpublished manuscript, June 2006.

[14] G. Neglia, G. Calbi, D. Towsley, and G. Vardoyan, “The Role of Network Topology for Distributed Machine Learning,” *Proceedings of IEEE INFOCOM*, pp. 2350–2358, 2019.

[15] T. Vogels, H. Hendrikx, and M. Jaggi, “Beyond Spectral Gap: The Role of the Topology in Decentralized Learning,” *Advances in Neural Information Processing Systems*, vol. 35, 2022.

[16] A. Ioannou, “Dynamic Topology Optimization for Non-IID Data in Decentralized Learning,” Bachelor’s thesis, Delft University of Technology, 2025.

[17] H. Xiao, K. Rasul, and R. Vollgraf, “Fashion-MNIST: A Novel Image Dataset for Benchmarking Machine Learning Algorithms,” arXiv:1708.07747, 2017.
