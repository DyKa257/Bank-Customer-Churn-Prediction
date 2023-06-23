import pandas as pd 
import numpy as np
import mlflow
import shap

# For visualization
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
# Makes sure we see all columns
pd.set_option('display.max_columns', None)
from sklearn.model_selection import train_test_split

# Fit models
from xgboost import XGBClassifier

class DataLoader():
    def __init__(self):
        self.data = None

    def load_dataset(self, path="data\Churn_Modelling.csv"):
        self.data = pd.read_csv(path)

    
    def exploratory_data_analysis(self):
        labels = 'Exited', 'Retained'
        sizes = [self.data.Exited[self.data['Exited']==1].count(), self.data.Exited[self.data['Exited']==0].count()]
        explode = (0, 0.1)
        fig1, ax1 = plt.subplots(figsize=(10, 8))
        ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
                shadow=True, startangle=90)
        ax1.axis('equal')
        plt.title("Proportion of customer churned and retained", size = 20)
        plt.show()

        # We first review the 'Status' relation with categorical variables
        fig, axarr = plt.subplots(2, 2, figsize=(20, 12))
        sns.countplot(x='Geography', hue = 'Exited',data = self.data, ax=axarr[0][0])
        sns.countplot(x='Gender', hue = 'Exited',data = self.data, ax=axarr[0][1])
        sns.countplot(x='HasCrCard', hue = 'Exited',data = self.data, ax=axarr[1][0])
        sns.countplot(x='IsActiveMember', hue = 'Exited',data = self.data, ax=axarr[1][1])
        fig.suptitle("The 'Status' relation with categorical variables", size=40)

        # Relations based on the continuous data attributes
        fig, axarr = plt.subplots(3, 2, figsize=(20, 12))
        sns.boxplot(y='CreditScore',x = 'Exited', hue = 'Exited',data = self.data, ax=axarr[0][0])
        sns.boxplot(y='Age',x = 'Exited', hue = 'Exited',data = self.data, ax=axarr[0][1])
        sns.boxplot(y='Tenure',x = 'Exited', hue = 'Exited',data = self.data, ax=axarr[1][0])
        sns.boxplot(y='Balance',x = 'Exited', hue = 'Exited',data = self.data, ax=axarr[1][1])
        sns.boxplot(y='NumOfProducts',x = 'Exited', hue = 'Exited',data = self.data, ax=axarr[2][0])
        sns.boxplot(y='EstimatedSalary',x = 'Exited', hue = 'Exited',data = self.data, ax=axarr[2][1])
        fig.suptitle("Relations based on the continuous data attributes", size=40)

    def feature_engineering(self):
        # Feature engineering
        self.data['BalanceSalaryRatio'] = self.data.Balance/self.data.EstimatedSalary
        self.data['TenureByAge'] = self.data.Tenure/(self.data.Age)
        self.data['CreditScoreGivenAge'] = self.data.CreditScore/(self.data.Age)

    def preprocess_data(self):
        # Drop id as it is not relevant
        self.data.drop(["RowNumber", "CustomerId", "Surname"], axis=1, inplace=True)
        
        # Arrange columns by data type for easier manipulation
        continuous_vars = ['CreditScore',  'Age', 'Tenure', 'Balance','NumOfProducts', 'EstimatedSalary', 'BalanceSalaryRatio',
                        'TenureByAge','CreditScoreGivenAge']
        cat_vars = ['HasCrCard', 'IsActiveMember','Geography', 'Gender']
        self.data = self.data[cat_vars + continuous_vars + ['Exited']]

        # One-hot encode all categorical columns
        categorical_cols = ["Geography",
                            "Gender"]
        encoded = pd.get_dummies(self.data[categorical_cols], 
                                prefix=categorical_cols)
        encoded = encoded.astype(int)

        # Update data with new columns
        self.data = pd.concat([encoded, self.data], axis=1)
        self.data.drop(categorical_cols, axis=1, inplace=True)

        # minMax scaling the continuous variables
        minVec = self.data[continuous_vars].min().copy()
        maxVec = self.data[continuous_vars].max().copy()
        self.data[continuous_vars] = (self.data[continuous_vars]-minVec)/(maxVec-minVec)
        
        # Standardization 
        # Usually we would standardize here and convert it back later
        # But for simplification we will not standardize / normalize the features

    def get_data_split(self):
        X = self.data.iloc[:,:-1]
        y = self.data.iloc[:,-1]
        return train_test_split(X, y, test_size=0.20, random_state=200)
    
# Fit Extreme Gradient Boost Classifier
def XGBClassifier_model(X_train, y_train):
    mlflow.autolog()
    XGB = XGBClassifier(base_score=None, booster=None, callbacks=None,
              colsample_bylevel=None, colsample_bynode=None,
              colsample_bytree=None, early_stopping_rounds=None,
              enable_categorical=False, eval_metric=None, feature_types=None,
              gamma=0.001, gpu_id=None, grow_policy=None, importance_type=None,
              interaction_constraints=None, learning_rate=0.2, max_bin=None,
              max_cat_threshold=None, max_cat_to_onehot=None,
              max_delta_step=None, max_depth=7, max_leaves=None,
              min_child_weight=1, missing=np.nan, monotone_constraints=None,
              n_estimators=5, n_jobs=None, num_parallel_tree=None,
              predictor=None, random_state=None)
    XGB.fit(X_train, y_train)
    return XGB

def explainable_AI(model, X_test):
    # explain the model's predictions using SHAP
    # (same syntax works for LightGBM, CatBoost, scikit-learn, transformers, Spark, etc.)
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)
    
    # visualize the first prediction's explanation with a force plot
    shap.initjs()
    force_plot = shap.plots.force(shap_values[0])
    display(force_plot)
    # Feature summary
    shap.plots.bar(shap_values)
