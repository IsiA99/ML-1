# Car Engine Risk Predicting Model

This project aims to predict the risk associated with car engines using the 'Automobile' dataset. The project involves data preprocessing, feature engineering, and model training using various machine learning algorithms.

## Installation

To install the required packages, run the following command:

```sh
pip install -r requirements.txt
```

## Dataset

The dataset used in this project is the 'Automobile' dataset, which can be found [here](https://archive.ics.uci.edu/static/public/10/data.csv).

## Data Preprocessing

1. **Handling Missing Values**:
   - Mean imputation for continuous features.
   - Mode imputation for categorical features.

2. **Encoding Categorical Variables**:
   - One-hot encoding for categorical features.
   - Label encoding for binary features.

3. **Normalizing Continuous Features**:
   - Using `StandardScaler` to normalize continuous features.

4. **Handling Outliers**:
   - Clipping values within the interquartile range.

## Feature Engineering

A new feature `engine_risk` is created using the formula:

```python
df['engine_risk'] = (df['engine-size'] * df['num-of-cylinders']) / df['curb-weight']
```

## Model Training

The following machine learning models are used to predict `engine_risk`:

- RandomForestRegressor
- GradientBoostingRegressor
- LGBMRegressor

## Evaluation

Model performance is evaluated using metrics such as mean squared error and R² score.

## Usage

To run the notebook, open `204084T_ML Assignment .ipynb` in Jupyter Notebook or JupyterLab and execute the cells.

## Requirements

- pandas
- numpy
- scikit-learn
- lightgbm
- matplotlib
- seaborn
