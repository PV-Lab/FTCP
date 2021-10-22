# Fourier-Transformed Crystal Properties (FTCP)

This software package implements Fourier-Transformed Crystal Properties (`FTCP`) that is an invertible crystallographic representation, and its associative variational autoencoder (VAE) to inversely design new crystals based on material properties, spanning various chemical compositions and crystal structures (achieving **_general_** inverse design).

The package provides functions in three major aspects:
- **FTCP Representation**: Represent crystals using FTCP representation (in `data.py`)
- **VAE Model with Property-Structured Latent Space**: Encode represented crystals into a property-structured latent space using VAE,  with a target-learning branch to achieve "property-structure" (in `model.py`)
- **Sampling (Inverse Design)**: Sample the material-semantic latent space according to design target(s) for new crystals (in `sampling.py`)

The following paper describes the details of the FTCP representation and framework: 
[Inverse design of crystals using generalized invertible crystallographic
representation](https://arxiv.org/pdf/2005.07609.pdf)

# Table of Contents
- [Fourier-Transformed Crystal Properties (FTCP)](#fourier-transformed-crystal-properties-ftcp)
- [How to Cite](#how-to-cite)
- [Installation](#installation)
- [Usage](#usage)
- [Authors](#authors)

# How to Cite

Please cite the following work if you want to use FTCP.
```
@misc{ren2020inverse,
  title = {Inverse design of crystals using generalized invertible crystallographic representation},
  author = {Ren, Zekun and Tian, Siyu Isaac Parker and Noh, Juhwan and Oviedo, Felipe and Xing, Guangzong and Li, Jiali and Liang, Qiaohao and Zhu, Ruiming and Aberle, Armin G. and Sun, Shijing and Wang, Xiaonan and Liu, Yi and Li, Qianxiao and Senthilnath, Jayavelu and Hippalgaonkar, Kedar and Jung, Yousung and Buonassisi, Tonio},
  year = {2020},
  eprint={2005.07609v3},
  archivePrefix={arXiv},
  primaryClass={physics.comp-ph}
}
```

# Installation

To install, clone the repository, navigate to the folder, and use
`pip install -r requirement.txt`


# Usage

Run `main.py` for ground-state properties, and run `main_semi.py` for thermoelectric power factor design. The two main scripts go through a typical FTCP inverse design pipeline, where (1) crystals are represented using FTCP representation, (2) a property-structured latent space is obtained by training the VAE + target-learning branch model, and (3) new crystals are designed by sampling the latent space with decoding, and postprocessing.

The package contains the following module and scripts:

| Module | Description |
| ------------- | ------------------------------ |
| `main.py`      | Whole inverse design pipeline (Design `Case 2` in paper)|
| `main_semi.py`      | Semi-supervised-learning inverse design pipeline for thermoelectric power factor (excited-state property) designs (Design `Case 3` in paper)|
| `data.py`  | Data-related functions, including data retrieval from [Materials Project](https://materialsproject.org/), and FTCP representation|
| `model.py`  | Model-related functions, including setting up the VAE + target-learning branch model|
| `sampling.py`  | Sampling-related functions, including getting the chemical formulas, checking for compositional uniqueness, and outputting CIFs of designed crystals|
| `ultils.py` | Script for auxiliary functions|
| `requirements.txt`| required packages|

# Authors

The code was primarily written by Zekun Ren and Siyu Isaac Parker Tian, under supervision of Prof. Tonio Buonassisi.
