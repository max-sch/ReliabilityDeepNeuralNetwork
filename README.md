# Quantifying Lower Reliability Bounds of Deep Neural Networks

This repository contains code for the analysis of deep neural networks (DNNs). Specifically, it implements a procedure for determining lower reliability bounds of DNNs using the conformal prediction framework. Currently, only classification models are supported, but an extension to regression models is planned.


## Installation

For installation, clone the repository and install the required dependencies:

```
git clone ...
cd ReliabilityDeepNeuralNetwork
python -m venv env # Activate after installation
pip install -r requirements.txt
```

## Experiments

To execute the experiments for models trained on the MNIST, Fashion MNIST, and CIFAR-10 datasets, simply run `python main.py`.

Note: Running experiments for models trained on the HAM10000 dataset requires additional setup and installations. Detailed documentation for these experiments will be provided separately.

## Citation
This repository provides the code of our paper: Quantifying Lower Reliability Bounds of Deep Neural Networks. The details of the approach are explained in the paper. If you find this repository helpful, please consider citing our paper:
```
@inproceedings{scheerer2024quantifying,
  title={Quantifying Lower Reliability Bounds of Deep Neural Networks},
  author={Scheerer, Max and Take, Marius and Klamroth, Jonas},
  booktitle={2024 IEEE 35th International Symposium on Software Reliability Engineering Workshops (ISSREW)},
  pages={247--254},
  year={2024},
  organization={IEEE}
}
```
