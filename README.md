# CIFAR-10 Image Classification with Pruning in Convolutional Neural Networks (CNNs)

This repository contains code implementing a CNN-based classifier for the CIFAR-10 dataset. The model architecture includes convolutional and fully connected layers. Additionally, it incorporates a pruning mechanism inspired by Lottery Ticket Hypothesis to achieve sparsity in the network.

## Overview

The code provided here involves:

- Preprocessing CIFAR-10 dataset and setting up data loaders for training and testing.
- Implementing a CNN-based model architecture named `Network` that consists of convolutional and fully connected layers.
- Training the model using a specified number of epochs, rounds, and sparsity levels.
- Implementing a pruning mechanism to prune weights from convolutional and linear layers based on the Lottery Ticket Hypothesis.
- Evaluating the accuracy of the model before and after pruning, with and without retraining.

## Prerequisites

- Python 3.x
- PyTorch
- Matplotlib
- NumPy
- scikit-learn

## Model Architecture
The Network class represents the CNN architecture. It comprises convolutional layers followed by fully connected layers.

## Pruning Mechanism
The code implements a pruning mechanism inspired by the Lottery Ticket Hypothesis. The Lottery Ticket Hypothesis, proposed by Jonathan Frankle and Michael Carbin, posits that within large neural networks, there exist sparse subnetworks (winning tickets) that can match the performance of the original network when trained in isolation. These subnetworks can be uncovered by iterative pruning and retraining of the network's weights.

### Key Points:

Sparse Subnetworks: Despite the over-parameterization in deep neural networks, there exist small, trainable subnetworks that achieve comparable performance.
Iterative Pruning: The process involves iteratively pruning connections or weights based on their magnitudes and then retraining the pruned network to recover performance.
Transferability: These winning tickets or sparse subnetworks can be transferred across different tasks and architectures.
Understanding this hypothesis aids in exploring strategies for network compression, acceleration, and understanding the underlying structure of deep learning models.

## Testing
The code evaluates the model accuracy before and after pruning, with and without retraining, on the CIFAR-10 test dataset.

## References
This implementation incorporates ideas from the Lottery Ticket Hypothesis for CNN pruning. For more information, refer to the original paper:

[The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks](https://arxiv.org/abs/1803.03635) by Jonathan Frankle and Michael Carbin.
