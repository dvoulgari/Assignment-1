# Machine Learning for Computational Biology: BMI Prediction Project
This repository contains the report and code for a machine learning project focused on predicting Body Mass Index (BMI) using metagenomic data from gut bacteria.

## Project Objective
The main goal of this project is to develop and evaluate a regression model that can accurately predict BMI using data on gut bacteria composition.

## Methodology & Models
The project follows a structured workflow that includes:

1) **Data Exploration and Preprocessing**: Handling missing values, encoding categorical variables like Sex, and scaling numerical features such as host age and bacterial abundances.

2) **Model Development**: Training and evaluating three baseline models: Elastic Net, Support Vector Regression (SVR), and Bayesian Ridge.

3) **Feature Selection**: Applying different feature selection strategies to each model to improve interpretability and performance. This includes Permutation Importance for SVR and Recursive Feature Elimination (RFE) for Elastic Net and Bayesian Ridge.

4) **Hyperparameter Tuning**: Optimizing model performance using a grid search approach with cross-validation.


## Key Findings
- All models demonstrated limited explanatory power, with R² values below 0.24, highlighting the inherent challenges of predicting BMI from microbiome data alone.

- The SVR model was selected as the "winner," achieving the best overall performance with the lowest RMSE and MAE.

- Different models selected partially overlapping feature sets, suggesting that the signal is diffuse across many bacterial species rather than concentrated in a few.


## Repository Structure
- `data/`: Contains the datasets used for the project.

- `models/`: Stores the trained machine learning models.

- `notebooks/`: Includes the Jupyter notebooks for data analysis, model development, and evaluation.

- `src/`: Holds source code files with reusable functions.

- `requirements.txt`: Lists all necessary Python dependencies.

## How to use

1) Clone the repository:
```bash
git clone https://github.com/dvoulgari/Assignment-1.git
```

2) Navigate to the directory and install dependencies:

cd Assignment-1
pip install -r requirements.txt

3) Launch the Jupyter Notebook to explore the code and project files:

jupyter notebook
