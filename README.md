# Implementation of a Neural Network from scratch

## Description
This project implements a neural network from scratch using Python, focusing on understanding the internal workings of a neural network model without relying on high-level machine learning libraries.

## How to use
### 1. Install Dependencies
To get started, you need to install the following dependencies:

- **Numpy**: A library used for scientific computing in Python.
- **Pandas**: A library for data manipulation and analysis.
- **Matplotlib**: A library for creating and displaying plots and graphs.
- **Scikit-learn (Sklearn)**: A library that provides functions for data preprocessing.
- **Seaborn**: A library for creating and displaying statistical graphics.
  
You can install these dependencies using the `requirements.txt` file with the following command:

```bash
pip install -r requirements.txt
```
### 2. Running the Script
The script can be run in two modes: crossvalidation and test. Below are the instructions for each mode.

**Cross-validation Mode**
This mode performs cross-validation on the training data to find the best model.

Run the script with:
```bash
python <dataset_name>.py crossvalidation
```
In cross-validation mode, you can adjust the following parameters to explore different model combinations:

**Layer Combinations**: Defines the number of neurons in the hidden layers. 

Example: ```layer_combinations = [[64, 32], [32, 16], [16, 8]] ```

**Regularizers**: Specifies the type of regularization to use.

Example: ```regularizers = ["l2", "l1", None] ```

**Optimizers**: Lists the optimizers to test during training.

Example: ```optimizers = ["adam", "rmsprop", "sgd_momentum"] ```

**Dropout**: Indicates whether dropout should be applied for regularization.

Example: ```dropout = [True, False] ```

**Activation Functions**: Sets the activation functions used in the network layers.

Example: ```activation_functions = ["relu", "tanh"]```

These parameters can be modified directly within the train_and_validate function to test different network architectures and training configurations.

**Test Mode**
This mode retrains the model on the training data and evaluates it on the test set.

Run the script with:
```bash
python <dataset_name>.py test
```
