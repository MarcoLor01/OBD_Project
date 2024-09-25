# Implementation of a Neural Network from scratch

## Description

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

**Test Mode**
This mode retrains the model on the training data and evaluates it on the test set.

Run the script with:
```bash
python <dataset_name>.py test
```
