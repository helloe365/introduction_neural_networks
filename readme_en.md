# Project: Introduction to Neural Networks

- 中文: [readme.md](readme.md)

Overview
----
This repository is an educational collection of neural network examples and utilities, covering numerical differentiation, two-layer networks, convolutional networks (including PyTorch implementations), and training scripts. It is intended for learning and hands-on practice.

Contents (high level)
----
- `check_gpu.py`: simple GPU check script.
- `common/`: core implementation modules (layers, optimizers, trainer, utils).
- `dataset/`: data loaders and MNIST helper.
- `MNIST/`: many MNIST examples including CNNs, ResNet, training and test scripts.
- `basic_principle/`: basic principle examples such as gradient computations and activation function comparisons.

Environment & Dependencies
----
- Python 3.8+ recommended.
- Typical dependencies (create `requirements.txt` accordingly):

```bash
numpy
matplotlib
torch   # if running PyTorch examples
torchvision
pillow
```

Quick Start
----
1. Clone the repository.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run a quick check:

```bash
python check_gpu.py
```

Run an example training script:

```bash
python MNIST/卷积神经网络/CNN/train_convnet.py
```

Notes
----
- The repository contains multiple implementation styles (NumPy hand-rolled implementations and PyTorch versions) for comparative learning.
- To reproduce experiments reliably, fix random seeds, record hyperparameters, and save model weights (many training scripts include save logic).

Contributing & Contact
----
Issues and PRs are welcome to improve examples, fix bugs, or add tutorials. When filing an issue, include your environment and a minimal reproduction.

License
----
Please confirm licensing with the repository owner prior to redistribution; no explicit LICENSE file is included by default.
