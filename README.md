# Fourier-Transformed Crystal Properties (FTCP)

This software package implements Fourier-Transformed Crystal Properties (`FTCP`) that is an invertible crystallographic representation, and its associative variational autoencoder (VAE) to inversely design new crystals based on material properties, spanning various chemical compositions and crystal structures (achieving **_general_** inverse design).

The package provides three major functions:
- Represent crystals using FTCP representation (in `data.py`)
- Encode represented crystals into a property-structured latent space using VAE (co-trained with  a target-learning branch to achieve "property-structure") (in `model.py`)
- Sample the material-semantic latent space according to design target(s) for new crystals (in `sampling.py`)

The following paper describes the details of the FTCP representation and framework:
[Inverse design of crystals using generalized invertible crystallographic
representation] (https://arxiv.org/pdf/2005.07609.pdf)

# Table of Contents
- [Fourier-Transformed Crystal Properties (FTCP)](#fourier-transformed-crystal-properties-ftcp)
- [Table of Contents](#table-of-contents)
- [Installation](#installation)
- [Usage](#usage)
- [Authors](#authors)

# How to Cite

Please cite the following work if you want to use FTCP.
```
@article{PhysRevLett.120.145301,
  title = {Crystal Graph Convolutional Neural Networks for an Accurate and Interpretable Prediction of Material Properties},
  author = {Xie, Tian and Grossman, Jeffrey C.},
  journal = {Phys. Rev. Lett.},
  volume = {120},
  issue = {14},
  pages = {145301},
  numpages = {6},
  year = {2018},
  month = {Apr},
  publisher = {American Physical Society},
  doi = {10.1103/PhysRevLett.120.145301},
  url = {https://link.aps.org/doi/10.1103/PhysRevLett.120.145301}

@article{Wang2021crabnet,
 author = {Wang, Anthony Yu-Tung and Kauwe, Steven K. and Murdock, Ryan J. and Sparks, Taylor D.},
 year = {2021},
 title = {Compositionally restricted attention-based network for materials property predictions},
 pages = {77},
 volume = {7},
 number = {1},
 doi = {10.1038/s41524-021-00545-1},
 publisher = {{Nature Publishing Group}},
 shortjournal = {npj Comput. Mater.},
 journal = {npj Computational Materials}
}
}
```

# Installation

To install, clone the repository, navigate to the folder, and use:
`pip install -r requirement.txt`


# Usage

run `CVAE.py` with given number of elements and nsites.  This generates the crystal representation in both real and momentum space and encodes such representation into VAE latent space.

The package contains the following module and scripts:

| Module | Description |
| ------------- | ------------------------------ |
| `CVAE.py`      | Script for training the VAE with structured latent space according to material properties      |
| `featurizer.py`  | Script for data mining 3D crystal structure from MP.org and featurize it into both real and momentum space representation|
| `ultils.py` | Script for auxiliary functions|
| `requirements.txt`      | required packages    |

# Authors

The code was primarily written by Zekun Ren, and Siyu Isaac Parker Tian, under supervision of Prof. Tonio Buonassisi.
