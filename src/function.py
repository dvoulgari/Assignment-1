import numpy as np
import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import KFold, cross_val_score, train_test_split, GridSearchCV, ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNet, BayesianRidge
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import RFECV
from sklearn.inspection import permutation_importance
from sklearn.base import clone


# Load datasets
def load_data (file_path):
    df=pd.read_csv(file_path)
    return df

class DataHandler:
    def __init__(self, file_path=None, df=None):
        self.file_path = file_path
        self.df = df if df is not None else self.load_data()
        
    def load_data(self):
        """Load data from file path"""
        return pd.read_csv(self.file_path)
    
    def summarize_data(self, dataset_name=""):
        """Basic data summary"""
        print(f"\nEDA for {dataset_name} dataset:")
        print("Dataset shape:", self.df.shape)
        print("\nMissing values:\n", self.df.isnull().sum())
        print("\nDuplicates:", self.df.duplicated().sum())
        return self
    
    def categorical_summary(self):
        """Summary of categorical variables"""
        cat_cols = self.df.select_dtypes(include=['object']).columns
        print("\nCategorical Columns Summary:\n", self.df[cat_cols].describe())
        return self
    
    def visualize(self, df, target_column, dataset_name=""):
        """All visualization methods"""
        # Visualize the distribution of the target variable
        plt.figure(figsize=(10, 6))
        sns.histplot(df[target_column], bins=20, kde=True)
        plt.title(f"Distribution of {target_column} - {dataset_name}")
        plt.xlabel(target_column)
        plt.ylabel("Frequency")
        plt.show()

        # Visualize the distribution of sex variable
        plt.figure(figsize=(10, 6))
        sns.countplot(x=df["Sex"])
        plt.title("Distribution of Sex")
        plt.show()

        # Visualize the distribution of age variable
        plt.figure(figsize=(10, 6))
        sns.histplot(df["Host age"], bins=20, kde=True)
        plt.title(f"Distribution of Age - {dataset_name}")
        plt.xlabel("Host Age")
        plt.ylabel("Frequency")
        plt.show()

        # Plot a pairplot of the BMI and Host age columns
        sns.pairplot(df, vars=['Host age', 'BMI'], hue='Sex')
        plt.show()

        # Visualize the correlation matrix
        numeric_df = df.select_dtypes(include=['float64', 'int64'])  # Select only numeric columns
        if numeric_df.empty:
            print("No numeric columns available for correlation matrix.")
            return
        plt.figure(figsize=(10, 6))
        sns.heatmap(numeric_df.corr(), cmap='coolwarm', vmax=1, vmin=-1, annot=False)
        plt.title(f"'Correlation Between Bacterial Features - {dataset_name}")
        plt.show()

        # Plot correlation with the target column
        correlations = numeric_df.corr()[target_column].sort_values(ascending=False)
        plt.figure(figsize=(12, 6))
        sns.barplot(x=correlations.index, y=correlations.values, hue=correlations.index, dodge=False, palette="viridis", legend=False)
        plt.title(f"Correlation with {target_column} - {dataset_name}", fontsize=16)
        plt.xlabel("Features", fontsize=12)
        plt.ylabel("Correlation", fontsize=12)
        plt.xticks(rotation=90, fontsize=6)
        plt.tight_layout()
        plt.show()

        # Plot correlation with "Host age"
        correlations_age = numeric_df.corr()["Host age"].sort_values(ascending=False)
        plt.figure(figsize=(12, 6))
        sns.barplot(x=correlations_age.index, y=correlations_age.values,hue=correlations.index, dodge=False, palette="viridis", legend=False)
        plt.title(f"Correlation with Host Age - {dataset_name}", fontsize=16)
        plt.xlabel("Features", fontsize=12)
        plt.ylabel("Correlation", fontsize=12)
        plt.xticks(rotation=90, fontsize=6)
        plt.tight_layout()
        plt.show()
    
    def eda(self, dataset_name="", target_column=""):
        """Full EDA pipeline"""
        self.summarize_data(dataset_name)
        self.categorical_summary()
        self.visualize(self.df, target_column, dataset_name)
        return self

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------

# Function to extract features and target variable from the DataFrame and split them into training and testing sets
def feature_extraction_and_split(df, target_column="BMI", test_size=0.2, random_state=42):
    X = df.drop(columns=[target_column], axis=1)
    y = df[target_column]
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    print(f"Training set shape: {X_train.shape}, Testing set shape: {X_test.shape}")
    
    return {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}

class DataPreprocessor:
    def __init__(self):
        self.preprocessor = None
    
    def fit(self, X_train, y=None):
        # Identify feature types
        num_features = X_train.select_dtypes(include=["int64", "float64"]).columns
        cat_features = X_train.select_dtypes(include=["object", "category"]).columns
        
        # Configure binary encoding for sex column
        binary_encoder = OrdinalEncoder(
            categories=[['Female', 'Male']],
            dtype=int
        )
        
        # Create pipelines
        transformers = [
            ('num', StandardScaler(), num_features),
            ('Sex', Pipeline([
                ('encoder', binary_encoder)
            ]), ['Sex'])  # Apply only to sex column
        ]
        
        self.preprocessor = ColumnTransformer(
            transformers,
            remainder="drop",
            verbose_feature_names_out=False)  # Disable prefixes
        
        self.preprocessor.fit(X_train)
        return self
    
    def transform(self, X):
        return pd.DataFrame(
            self.preprocessor.transform(X),
            columns=self.preprocessor.get_feature_names_out()
        )

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
class ManualFeatureSelector():
    def __init__(self, selected_indices):
        self.selected_indices = selected_indices

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if hasattr(X, "iloc"):
            return X.iloc[:, self.selected_indices]
        else:
            return X[:, self.selected_indices]

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Create a class for training and evaluating the baseline models
class ModelTraining:
    def __init__(self, X_train, y_train, X_test, y_test, X_eval, y_eval):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.X_eval = X_eval
        self.y_eval = y_eval
        self.models = {
            "ElasticNet": ElasticNet(),
            "SVR": SVR(),
            "BayesianRidge": BayesianRidge()
        }
        self.results = {}

    # ================================================
    # TRAINING
    # ================================================
    def train_baseline_models(self):
        for name, model in self.models.items():
            print(f"\nTraining {name}...")

            # Train the model on the training data
            model.fit(self.X_train, self.y_train)

            # Predict on training data
            y_train_pred = model.predict(self.X_train)

            # Predict on test data
            y_test_pred = model.predict(self.X_test)

            # Predict on test data
            y_eval_pred = model.predict(self.X_eval)

            # Store results
            self.results[name] = {
                "model": model,
                "train_metrics": {
                    "rmse": np.sqrt(mean_squared_error(self.y_train, y_train_pred)),
                    "mae": mean_absolute_error(self.y_train, y_train_pred),
                    "r2": r2_score(self.y_train, y_train_pred)
                },
                "test_metrics": {
                    "rmse": np.sqrt(mean_squared_error(self.y_test, y_test_pred)),
                    "mae": mean_absolute_error(self.y_test, y_test_pred),
                    "r2": r2_score(self.y_test, y_test_pred)
                },
                "validation_metrics": {
                    "rmse": np.sqrt(mean_squared_error(self.y_eval, y_eval_pred)),
                    "mae": mean_absolute_error(self.y_eval, y_eval_pred),
                    "r2": r2_score(self.y_eval, y_eval_pred)
                },
            }

            # Print metrics for the model
            print(f"\n{name} - Training Metrics:")
            print(f"RMSE: {self.results[name]['train_metrics']['rmse']:.4f}")
            print(f"MAE: {self.results[name]['train_metrics']['mae']:.4f}")
            print(f"R²: {self.results[name]['train_metrics']['r2']:.4f}")

            print(f"\n{name} - Test Metrics:")
            print(f"RMSE: {self.results[name]['test_metrics']['rmse']:.4f}")
            print(f"MAE: {self.results[name]['test_metrics']['mae']:.4f}")
            print(f"R²: {self.results[name]['test_metrics']['r2']:.4f}")

            print(f"\n{name} - Validation Metrics:")
            print(f"RMSE: {self.results[name]['validation_metrics']['rmse']:.4f}")
            print(f"MAE: {self.results[name]['validation_metrics']['mae']:.4f}")
            print(f"R²: {self.results[name]['validation_metrics']['r2']:.4f}")

    
    def save_model(self, model_name):
        """Save the trained model to a file"""
        model = self.results[model_name]["model"]
        file_path = f"{model_name}_model.pkl"
        joblib.dump(model, file_path)
        print(f"Model saved to {file_path}")
        return file_path

    # ================================================
    # VISUALIZATION METHODS
    # ================================================
    def _plot_actual_vs_predicted(self, y_true, y_pred, model_name):
        """Visualize prediction accuracy"""
        plt.figure(figsize=(8, 6))
        ax = sns.scatterplot(x=y_true, y=y_pred, alpha=0.6)
        plt.plot([y_true.min(), y_true.max()], 
                [y_true.min(), y_true.max()], 
                'r--', lw=2)
        plt.title(f"{model_name}: Actual vs Predicted\nR² = {r2_score(y_true, y_pred):.3f}")
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def _plot_residuals(self, y_true, y_pred, model_name):
        """Analyze prediction errors"""
        residuals = y_true - y_pred
        plt.figure(figsize=(8, 6))
        sns.histplot(residuals, bins=30, kde=True)
        plt.axvline(0, color='r', linestyle='--')
        plt.title(f"{model_name}: Residual Distribution\n(MAE = {mean_absolute_error(y_true, y_pred):.3f})")
        plt.xlabel("Residuals")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
   
    # ================================================
    # MODEL COMPARISON METHODS
    # ================================================
    def _create_metrics_df(self):
        """Prepare metrics for comparison"""
        metrics = []
        for name, res in self.results.items():
            for stage in ["train", "test", "validation"]:
                metrics.append({
                    "Model": name,
                    "Stage": stage.capitalize(),
                    "RMSE": res[f"{stage}_metrics"]["rmse"],
                    "MAE": res[f"{stage}_metrics"]["mae"],
                    "R²": res[f"{stage}_metrics"]["r2"]
                })
        return pd.DataFrame(metrics)

    def compare_models(self):
        """RMSE comparison plot across all stages"""
        # Prepare RMSE data
        rmse_data = []
        for name, res in self.results.items():
            for stage in ['train', 'test', 'validation']:
                rmse_data.append({
                    'Model': name,
                    'Stage': stage.capitalize(),
                    'RMSE': res[f"{stage}_metrics"]["rmse"]
                })
        
        rmse_df = pd.DataFrame(rmse_data)
        
        # Create the plot
        plt.figure(figsize=(8, 5))
        sns.barplot(data=rmse_df, x='Model', y='RMSE', hue='Stage', 
                palette='viridis', alpha=0.8)
        
        # Customize the plot
        plt.title('RMSE Comparison - Baseline models', fontsize=12, pad=10)
        plt.xlabel('Model', fontsize=10)
        plt.ylabel('RMSE', fontsize=10)
        plt.legend(title='Stage', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Add value labels
        ax = plt.gca()
        for p in ax.patches:
            ax.annotate(f"{p.get_height():.2f}", 
                    (p.get_x() + p.get_width()/2., p.get_height()),
                    ha='center', va='center', 
                    xytext=(0, 5), 
                    textcoords='offset points',
                    fontsize=8)
        
        plt.tight_layout()
        plt.show()

    def get_best_model(self, metric="rmse"):
        best_model = None
        best_score = float('inf') if metric in ["rmse", "mae"] else float('-inf')
        
        for name, result in self.results.items():
            score = result["validation_metrics"][metric]
            
            if (metric in ["rmse", "mae"] and score < best_score) or \
               (metric == "r2" and score > best_score):
                best_score = score
                best_model = name
                
        print(f"\nBest model by {metric}: {best_model} ({best_score:.4f})")
        return best_model, self.results[best_model]

    # ================================================
    # RUN ALL
    # ================================================

    def run_all(self):
        """Complete workflow: train, evaluate, compare, and save models"""
        print("="*50)
        print("TRAINING ALL BASELINE MODELS")
        print("="*50)
        
        self.train_baseline_models()
        
        print("\n" + "="*50)
        print("MODEL COMPARISON SUMMARY")
        print("="*50)
        self.compare_models()
        
        print("\n" + "="*50)
        print("BEST MODELS")
        print("="*50)
        self.get_best_model(metric="rmse")
        self.get_best_model(metric="mae")
        self.get_best_model(metric="r2")
        
        print("\n" + "="*50)
        print("SAVING MODELS")
        print("="*50)
        for model_name in self.models.keys():
            self.save_model(model_name)

        return self

#--------------------------------------------------------------------------------------------------------------------------------------

