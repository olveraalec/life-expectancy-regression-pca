# Life Expectancy Prediction Using Linear Regression and PCA

## Project Overview

This project predicts life expectancy using socioeconomic, mortality, health, and vaccination-related variables. The goal is to compare a selected-feature linear regression model against a PCA-based regression model, focusing on predictive accuracy, cross-validation stability, dimensionality reduction, and interpretability.

## Dataset

The dataset contains 2,928 observations and 22 variables. The target variable is `life_expectancy`.

The predictors include variables related to mortality, disease, economic development, healthcare expenditure, physical health indicators, vaccination rates, and country development status.

## Methods

This project compares two regression approaches:

1. **Multiple Linear Regression without PCA**
   - Uses selected predictors based on correlation with life expectancy.
   - Prioritizes interpretability and direct feature-level explanation.

2. **Linear Regression with PCA**
   - Uses principal components after preprocessing and scaling.
   - Prioritizes dimensionality reduction, stability, and reduced multicollinearity.

## Preprocessing Pipeline

The preprocessing pipeline included:

- cleaning column names
- separating predictors and target variable
- replacing certain zero values with missing values
- applying log transformations to skewed variables
- KNN imputation for missing values
- dummy encoding categorical variables
- removing high-cardinality country labels
- standardizing features
- optionally applying PCA

## Model Evaluation

Models were evaluated using:

- Root Mean Squared Error, RMSE
- R-squared, R²
- cross-validation
- train/test performance comparison

## Results

| Model | Setup | Test RMSE | Test R² | Main Advantage |
|---|---:|---:|---:|---|
| Linear Regression | 6 selected features | ~2.97 | ~0.905 | Interpretability and slightly better raw accuracy |
| PCA Regression | 3 principal components | ~3.13 | ~0.895 | Stability and dimensionality reduction |

## Visual Results

### Linear Regression Feature Sweep

The feature sweep compares model performance as the number of selected predictors increases.

![RMSE vs Number of Features](figures/rmse_vs_features.png)

![R² vs Number of Features](figures/r2_vs_features.png)

### PCA Component Sweep

The PCA sweep compares model performance as the number of principal components increases.

![RMSE vs PCA Components](figures/rmse_vs_pca.png)

![R² vs PCA Components](figures/r2_vs_pca.png)

## Key Findings

The selected-feature linear regression model achieved slightly stronger raw predictive performance. However, the PCA regression model provided a lower-dimensional and more stable representation of the data.

This suggests that many predictors in the dataset contain overlapping information, especially among health, mortality, and economic development variables. PCA was useful because it compressed correlated predictors into orthogonal components while retaining most of the predictive structure.

## Technologies Used

- Python
- Jupyter Notebook
- NumPy
- Pandas
- Matplotlib
- Seaborn
- scikit-learn

## Repository Structure

```text
life-expectancy-regression-pca/
│
├── README.md
├── life_expectancy_analysis.ipynb
├── requirements.txt
├── data/
│   └── life_expectancy.csv
├── figures/
│   ├── rmse_vs_features.png
│   ├── r2_vs_features.png
│   ├── rmse_vs_pca.png
│   └── r2_vs_pca.png
└── report/
    └── academic_report.pdf
```

## How to Run

Clone the repository, install the required packages, and open the notebook.

```bash
pip install -r requirements.txt
```

```bash
jupyter notebook life_expectancy_analysis.ipynb
```

Then run the notebook cells from top to bottom.

## Future Improvements

Future improvements could include:

- testing regularized regression models such as Ridge and Lasso
- comparing tree-based models such as Random Forest or Gradient Boosting
- adding nonlinear feature engineering
- creating a cleaner Python script version of the notebook
- deploying the final model as a simple prediction tool