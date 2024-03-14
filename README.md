# Deep Learning Assignment 1 

## Introduction
In this assignment, I tackled the problem of image classification using the Fashion MNIST dataset. I employed various deep learning techniques and optimization algorithms to train neural networks and achieve high accuracy in classifying different fashion items.

## Dataset
I utilized the Fashion MNIST dataset, which consists of 60,000 training images and 10,000 test images belonging to 10 different categories of clothing and accessories.

## Model Architecture
- **Input Layer:** 784 neurons (flattened images of size 28x28)
- **Hidden Layers:** Variable number of layers and neurons per layer
- **Output Layer:** 10 neurons (corresponding to the 10 fashion categories)
- **Activation Functions:** Sigmoid, ReLU, and Tanh
- **Loss Function:** Cross Entropy,MSE
- **Optimization Algorithms:** SGD, Momentum, Nestrov, RMSprop, Adam, and Nadam
- **Weight Initialization Techniques:** Random and Xavier

## Hyperparameter Tuning
I performed hyperparameter tuning using the Bayesian optimization method provided by the wandb sweep functionality. The following hyperparameters were optimized:
- Learning Rate
- Batch Size
- Number of Hidden Layers
- Hidden Layer Size
- Number of Epochs
- Weight Decay
## Neural Network Training Script
This Python script is designed to train a neural network with customizable parameters. It supports various options for dataset selection, model configuration, optimization algorithms, and more.
### Command Syntax

```bash
python train.py [-wp WAND_PROJECT] [-we WAND_ENTITY] [-d DATASET] [-e EPOCHS] [-b BATCH_SIZE] [-l LOSS] [-o OPTIMIZER] [-lr LEARNING_RATE] [-m MOMENTUM] [-beta BETA] [-beta1 BETA1] [-beta2 BETA2] [-eps EPSILON] [-w_d WEIGHT_DECAY] [-w_i WEIGHT_INIT] [-nhl NUM_LAYERS] [-sz HIDDEN_SIZE] [-a ACTIVATION]
```
### Example
```bash
python train.py -wp myproject -we myentity -d fashion_mnist -e 10 -b 32 -l cross_entropy -o adam -lr 0.001 -m 0.9 -beta 0.9 -beta1 0.9 -beta2 0.999 -eps 1e-8 -w_d 0.0005 -w_i Xavier -nhl 3 -sz 64 -a ReLU
```
## Results
The training process was tracked using Weights and Biases (wandb), allowing us to monitor various metrics such as training loss, training accuracy, validation loss, and validation accuracy. Through the sweep, I identified the best combination of hyperparameters that maximized the validation accuracy.<br>
Wandb Report Link :- https://wandb.ai/cs23m063/deep_learn_assignment_1/reports/CS6910-Assignment-1--Vmlldzo3MDg4NDQ2

## Conclusion
Through rigorous experimentation and hyperparameter tuning, I successfully trained neural networks with optimal configurations to classify fashion items with high accuracy. This assignment demonstrates the importance of hyperparameter optimization in achieving optimal performance in deep learning tasks.

