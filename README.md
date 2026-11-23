# Iris Classification using Gaussian Naive Bayes

A machine learning implementation demonstrating multi-class classification on the classic **Iris dataset** using **Gaussian Naive Bayes (`GaussianNB`)** in Python with Scikit-Learn.

## Overview

Because the measurements in the Iris dataset are continuous numerical values, **Gaussian Naive Bayes** is a natural probabilistic baseline. It assumes that the continuous values associated with each class are distributed according to a normal (Gaussian) distribution:

$$P(x_i \mid y) = \frac{1}{\sqrt{2\pi\sigma_y^2}} \exp\left(-\frac{(x_i - \mu_y)^2}{2\sigma_y^2}\right)$$

where $\mu_y$ and $\sigma_y^2$ represent the mean and variance of feature $x_i$ for class $y$.

## Dataset Details

* **Source:** `sklearn.datasets.load_iris`
* **Total Samples:** 150
* **Features (4 continuous):** Sepal length, Sepal width, Petal length, and Petal width (in cm)
* **Target Classes (3):**
  * `0` — Iris Setosa
  * `1` — Iris Versicolor
  * `2` — Iris Virginica

## Workflow

1. **Data Loading:** Loaded feature matrix $X$ and target vector $y$ using `load_iris(return_X_y=True)`.
2. **Sample Inference Check:** Verified initial model fitting and single-sample prediction on `[5.1, 3.5, 1.4, 0.2]`, correctly predicting class `0`.
3. **Train-Test Split:** Partitioned the dataset using a 70/30 split (`test_size=0.3`, `random_state=1`), yielding 105 training samples and 45 testing samples.
4. **Model Training:** Trained `GaussianNB()` on the training split.
5. **Model Evaluation:** Assessed generalization performance on the unseen test set using accuracy score, classification report, and confusion matrix.

## Results & Evaluation

The trained Gaussian Naive Bayes model achieved an overall test accuracy of **93.33%** (`0.9333`).

### Classification Report

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **0 (Setosa)** | 1.00 | 1.00 | 1.00 | 14 |
| **1 (Versicolor)** | 0.94 | 0.89 | 0.91 | 18 |
| **2 (Virginica)** | 0.86 | 0.92 | 0.89 | 13 |
| **Accuracy** | | | **0.93** | **45** |
| **Macro Avg** | 0.93 | 0.94 | 0.93 | 45 |
| **Weighted Avg** | 0.94 | 0.93 | 0.93 | 45 |

### Confusion Matrix

```text
[[14  0  0]
 [ 0 16  2]
 [ 0  1 12]]
```

* **Class 0** was classified with 100% precision and recall (14/14).
* **Classes 1 and 2** showed minor overlap, with 2 Versicolor samples predicted as Virginica and 1 Virginica sample predicted as Versicolor.

## Requirements & Usage

### Prerequisites

* Python 3.x
* `scikit-learn`
* Jupyter Notebook or Google Colab

### Running the Notebook

1. Clone this repository:
   ```bash
   git clone <your-repository-url>
   ```
2. Install Scikit-Learn if not already installed:
   ```bash
   pip install scikit-learn notebook
   ```
3. Launch Jupyter Notebook and open the `.ipynb` file to run all cells.
