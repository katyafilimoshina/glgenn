# Generalized Lipschitz Group Equivariant Neural Networks (GLGENN)

## Abstract
This repository contains implementation of Generalized Lipschitz Group Equivariant Neural Networks (GLGENN). These networks are equivariant to any pseudo-orthogonal transformation. The architecture of GLGENN contains
* ${C \kern -0.1em \ell}^{\overline{k}}_{p,q}$-linear layers,
* ${C \kern -0.1em \ell}^{\overline{k}}_{p,q}$-geometric product layers,
* ${C \kern -0.1em \ell}^{\overline{k}}_{p,q}$-normalization layers,

and employs theoretical results on generalized Lipschitz groups in Clifford algebras obtained in [1,2]. GLGENN generalize Clifford Group Equivariant Neural Networks presented in [3]

## Code Organization
* `algebra/`: Contains implementation of quaternion types subspaces in Clifford algebra.
* `data/`: Contains data loading scripts for experiments.
* `engineer/`: Contains training, evaluation, and visualization scripts.
* `experiments/`: Contains experiments on GLGENN.
* `layers/`: Contains architecture of GLGENN layers.
* `models/`: Contains models built from GLGENN layers.

## Experiments
* O(5) and O(7) regression tasks: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1iwVSXqOToJhCDZ56_bOAAVINKsBizbDj?usp=sharing)
* Equivariance check: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1UX_j6H-Dydbmnr3CbeU7ufFO7JTNdukZ?usp=sharing)

## References
[1] Filimoshina, E., Shirokov, D.: On generalization of Lipschitz groups and spin groups. Mathematical Methods in the Applied Sciences, 47(3), 1375--1400 (2024), [arXiv:2205.06045](https://arxiv.org/pdf/2205.06045)
\
[2] Shirokov, D.: On inner automorphisms preserving fixed subspaces of Clifford algebras. Adv. Appl. Clifford Algebras 31(30), (2021), [arXiv:2011.08287](https://arxiv.org/pdf/2011.08287)
\
[3] Ruhe, D., Brandstetter, J., Forré, P.: Clifford Group Equivariant Neural Networks (2023), [arXiv:2305.11141](https://arxiv.org/pdf/2305.11141)

