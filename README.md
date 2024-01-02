# Logistic Regression: A Comprehensive Overview

## Introduction

Logistic Regression is a popular statistical method used for binary classification. Unlike linear regression, which is used for continuous prediction tasks, logistic regression is specifically designed for predicting the probability of a binary outcome. It's widely used in various fields such as healthcare, finance, marketing, and more due to its simplicity and interpretability.

## Understanding Logistic Regression

### 1. Objective

The primary objective of logistic regression is to model the relationship between a set of independent variables (features) and a binary dependent variable (target) by estimating the probability that the given set of features belongs to a particular category.

### 2. Sigmoid Function

Logistic regression utilizes the sigmoid function (also known as the logistic function) to map predicted values to probabilities between 0 and 1. The formula for the sigmoid function is:

![Sigmoid Function](https://upload.wikimedia.org/wikipedia/commons/thumb/8/88/Logistic-curve.svg/640px-Logistic-curve.svg.png)

### 3. Hypothesis Representation

The hypothesis function in logistic regression is:

```
hθ(x) = 1 / (1 + e^(-θ^T * x))
```

Where:
- hθ(x) is the predicted probability that y = 1 given x and θ.
- θ is the parameter vector.
- x is the input feature vector.

### 4. Model Training

- **Cost Function**: The cost function for logistic regression is derived from the maximum likelihood estimation. It measures the difference between the predicted probability and the actual label.
- **Gradient Descent**: Optimization algorithms like gradient descent or its variations are used to minimize the cost function and update the model's parameters (θ) iteratively.

### 5. Decision Boundary

In logistic regression with two features, the decision boundary is a line that separates the two classes based on the predicted probabilities. For higher dimensions, it's a hyperplane.

### 6. Evaluation

- **Accuracy**: Measures the overall correctness of predictions.
- **Precision & Recall**: Indicates the model's performance on positive predictions and actual positives, respectively.
- **ROC Curve & AUC**: Receiver Operating Characteristic curve and Area Under the Curve show the trade-off between true positive rate and false positive rate.

## Advantages of Logistic Regression

- Simplicity and ease of implementation.
- Efficient for linearly separable data.
- Provides probabilities for outcomes.
- Less prone to overfitting.

## Limitations

- Assumes a linear relationship between features and the log-odds of the outcome.
- Can't handle non-linear relationships between features and target.
- Sensitive to outliers.

## Conclusion

Logistic Regression serves as a fundamental and powerful tool for binary classification tasks. Its simplicity, interpretability, and ability to provide probabilistic predictions make it a widely used algorithm across various domains.

---

# Titanic Survival Prediction Project

This project aims to predict survival outcomes for passengers on the Titanic using logistic regression. The model is trained on the historic Titanic dataset, considering various factors such as passenger class, gender, age, family relations onboard, and fare paid.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Usage](#usage)
- [Features](#features)
- [Model Training](#model-training)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview

The Titanic Survival Prediction Project utilizes logistic regression to predict whether a passenger would survive the Titanic disaster based on input features such as passenger class, gender, age, and more. The model is trained on historical data and can make survival predictions for new passenger details provided.

## Dataset

The dataset used for training and testing the model is sourced from [Kaggle](https://www.kaggle.com/c/titanic/data). It contains passenger information such as name, age, gender, ticket class, survival status, etc.

## Requirements

To run the project, you'll need:

- Python 3.x
- Pandas
- Scikit-learn
- Jupyter Notebook (for exploratory data analysis and model training)

Install required packages via:

```bash
pip install pandas scikit-learn jupyter
```

## Usage

1. Clone the repository:

```bash
git clone https://github.com/Kunal3012/titanic-survival-prediction.git
```

2. Navigate to the project directory:

```bash
cd titanic-survival-prediction
```

3. Run the prediction script:

```bash
python predict_survival.py
```

Follow the prompts to input passenger details and get the survival prediction.

## Features

The following features are considered for prediction:

- Passenger Class (Pclass)
- Gender (Sex)
- Age
- Number of Siblings/Spouses (SibSp)
- Number of Parents/Children (Parch)
- Fare

## Model Training

The logistic regression model is trained on the Titanic dataset using Python's scikit-learn library. The data is preprocessed, missing values handled, and categorical variables encoded for training.

## Results

The model achieved an accuracy of 77.09% on the test set, indicating its ability to predict survival outcomes based on the given features.

## Contributing

Contributions to improve the project are welcome! Feel free to submit issues and pull requests.

## License

This project is licensed under the [MIT License](LICENSE).

---
