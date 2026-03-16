# NysHD: Bridging the Gap between Hyperdimensional Computing and Kernel Methods via the Nyström Method [AAAI'25]

Official repository for the paper:

**Bridging the Gap between Hyperdimensional Computing and Kernel Methods via the Nyström Method**  
[[Paper]]([https://ojs.aaai.org/index.php/AAAI/article/view/34442])

```bibtex
@inproceedings{zhao2025bridging,
  title={Bridging the gap between hyperdimensional computing and kernel methods via the Nystr{\"o}m method},
  author={Zhao, Quanling and Thomas, Anthony Hitchcock and Brin, Ari and Yu, Xiaofan and Rosing, Tajana},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={21},
  pages={22813--22821},
  year={2025}
}
```

## Abstract
Hyperdimensional computing (HDC) is an approach from the cognitive science literature for solving information processing tasks using data represented as high-dimensional random vectors. The technique has a rigorous mathematical backing, and is easy to implement in energy-efficient and highly parallel hardware like FPGAs and “processing-in-memory” architectures. The effectiveness of HDC in machine learning largely depends on how raw data is mapped to a high-dimensional space. In this work, we propose NysHD, a new method for constructing this mapping that is based on the Nystrom method from the literature on kernel approximation. Our approach provides a simple recipe to turn any user-defined positive-semidefinite similarity function into an equivalent mapping in HDC. There is a vast literature on the design of such functions for learning problems. Our approach provides a mechanism to import them into the HDC setting, expanding the types of problems that can be tackled using HDC. Empirical evaluation against existing HDC encoding methods shows that NysHD can achieve, on average, 11% and 17% better classification accuracy on graph and string datasets, respectively.


## Repository Structure

```bash
.
├── ApproxAcc/                  # Normalized knerel matrix approximation against real Normalized knerel matrix
├── Framework/                  # HDC baselines and NysHD implementation on reported datasets
├── Sensitivity/                # Ablation study on hyperparameters
└── README.md



# To Run

```command line
python run_exp.py
```




































