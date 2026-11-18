# California Housing Price Prediction Using Neural Networks

A Deep Learning regression project that predicts median house values using a Neural Network built with **TensorFlow and Keras**.

The project uses the California Housing dataset and demonstrates a complete machine learning workflow, including data preprocessing, feature encoding, feature scaling, model training, validation, and performance evaluation.

## Project Overview

The objective of this project is to predict median house prices based on geographical and demographic features of California housing districts.

A Feedforward Neural Network (FNN) is implemented using TensorFlow's Sequential API to learn relationships between housing features and their corresponding median house values.

## Dataset

**File:** `housing.csv`

The dataset contains 20,640 records and 10 columns.

### Features

| Feature | Description |
|---|---|
| longitude | Geographical longitude |
| latitude | Geographical latitude |
| housing_median_age | Median age of houses |
| total_rooms | Total number of rooms |
| total_bedrooms | Total number of bedrooms |
| population | Population of the district |
| households | Number of households |
| median_income | Median household income |
| ocean_proximity | Proximity to the ocean |

**Target Variable:** `median_house_value`

## Technologies & Libraries

- Python
- NumPy
- Pandas
- Scikit-learn
- TensorFlow / Keras

## Project Workflow

### 1. Data Preprocessing
- Load the dataset using Pandas.
- Remove rows containing missing values.
- Separate input features (X) and target variable (y).
- Apply One-Hot Encoding to the `ocean_proximity` categorical feature.

### 2. Train, Validation & Test Split

The dataset is divided into three subsets:

| Dataset | Percentage |
|---|---:|
| Training | 80% |
| Validation | 10% |
| Testing | 10% |

A fixed random state (`42`) ensures reproducibility.

### 3. Feature Scaling

`StandardScaler` is used to standardize both input features and the target variable.

- Fit scalers using training data only.
- Apply the same transformations to validation and testing data.
- Use separate scalers for X and y.

### 4. Neural Network Architecture

The model uses a Feedforward Neural Network with three hidden layers.

| Layer | Neurons | Activation |
|---|---:|---|
| Input | 13 | — |
| Hidden Layer 1 | 64 | ReLU |
| Hidden Layer 2 | 32 | ReLU |
| Hidden Layer 3 | 16 | ReLU |
| Output | 1 | Linear |

### 5. Model Compilation

- **Optimizer:** Adam
- **Learning Rate:** 0.001
- **Loss Function:** Mean Squared Error (MSE)
- **Evaluation Metrics:** MAE and MSE
- **Batch Size:** 32
- **Maximum Epochs:** 200

### 6. Training Optimization

Two callbacks are implemented to improve the training process:

**Early Stopping**
- Monitors validation loss.
- Patience of 15 epochs.
- Restores the best model weights.

**ReduceLROnPlateau**
- Monitors validation loss.
- Reduces the learning rate by a factor of 0.5.
- Patience of 5 epochs.
- Minimum learning rate of 0.000001.

### 7. Model Evaluation

Predictions are transformed back to their original price scale before evaluation.

The following regression metrics are used:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

## Model Performance

The model achieved the following results on the test dataset:

| Evaluation Metric | Result |
|---|---:|
| MAE | 34,103.48 |
| MSE | 2,713,794,181.86 |
| RMSE | 52,094.09 |
| R² Score | 0.8028 |

**R² Score: 80.28%**

The model explains approximately 80.28% of the variance in the target variable on the test dataset.

## Project Structure

```text
California-Housing-Price-Prediction/
│
├── housing.csv
├── regression_nn.ipynb
└── README.md
```

## Installation & Usage

### 1. Clone the Repository

```bash
git clone <repository-url>
```

### 2. Navigate to the Project Directory

```bash
cd California-Housing-Price-Prediction
```

### 3. Install Dependencies

```bash
pip install numpy pandas scikit-learn tensorflow jupyter
```

### 4. Run the Notebook

```bash
jupyter notebook regression_nn.ipynb
```

Run the notebook cells sequentially to preprocess the dataset, train the neural network, and evaluate its performance.

## Learning Objectives

This project demonstrates practical implementation of:

- Neural Networks for regression problems.
- Data preprocessing and categorical encoding.
- Train, validation, and test splitting.
- Feature and target standardization.
- TensorFlow Sequential API.
- Model training and validation.
- Early Stopping and Learning Rate Scheduling.
- Regression evaluation metrics.

## Conclusion

This project demonstrates how a Feedforward Neural Network can be applied to a real-world regression problem.

By combining proper preprocessing, feature scaling, neural network architecture design, and training optimization techniques, the model achieves an R² score of approximately 0.80 on unseen test data.

---

**Built with Python, Scikit-learn, and TensorFlow.**
