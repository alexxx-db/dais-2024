# Databricks notebook source
# MAGIC %md 
# MAGIC
# MAGIC # MLflow
# MAGIC
# MAGIC
# MAGIC MLflow is a comprehensive framework for the complete machine learning lifecycle. MLflow provides tools for tracking experiments, packaging code into reproducible runs, sharing and deploying models, and evaluating models. 
# MAGIC
# MAGIC In this review notebook, **we will focus on tracking and logging components of MLflow**. 
# MAGIC
# MAGIC MLflow is pre-installed on the Databricks Runtime for ML.

# COMMAND ----------

from math import sqrt

import mlflow
import mlflow.data
import mlflow.sklearn
from mlflow.models.signature import infer_signature

from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_diabetes
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import sklearn

import pandas as pd

# COMMAND ----------

# Load and preprocess data
white_wine = pd.read_csv("/dbfs/databricks-datasets/wine-quality/winequality-white.csv", sep=';')
red_wine = pd.read_csv("/dbfs/databricks-datasets/wine-quality/winequality-red.csv", sep=';')
white_wine['is_red'] = 0.0
red_wine['is_red'] = 1.0
data_df = pd.concat([white_wine, red_wine], axis=0)

# Define classification labels based on the wine quality
data_labels = data_df['quality'] >= 7
data_df = data_df.drop(['quality'], axis=1)

# Split 80/20 train-test
X_train, X_test, y_train, y_test = train_test_split(
  data_df,
  data_labels,
  test_size=0.2,
  random_state=1
)

# COMMAND ----------

# MAGIC %md
# MAGIC # MLflow Tracking
# MAGIC MLflow tracking allows you to organize your machine learning training code, parameters, and models.

# COMMAND ----------

with mlflow.start_run(run_name="gradient_boost") as run:
  model = sklearn.ensemble.GradientBoostingClassifier(random_state=0)
  
  # log model paramters 
  model.fit(X_train, y_train)
  mlflow.log_params(model.get_params())

  predicted_probs = model.predict_proba(X_test)
  test_roc_auc = roc_auc_score(y_test, predicted_probs[:,1])
  
  # log evaluation metrics
  mlflow.log_metric("test_auc", test_roc_auc)
  print("Test AUC of: {}".format(test_roc_auc))

  signature = infer_signature(X_train, y_train)

  # log the model
  mlflow.sklearn.log_model(
    sk_model = model, 
    artifact_path="model-artifacts",
    signature=signature,
    registered_model_name="wine_quality_classifier")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Review the Model via the UI
# MAGIC
# MAGIC
# MAGIC To review the model and its details, follow these step-by-step instructions:
# MAGIC
# MAGIC + **Step 1: Go to the "Experiments" Section:**
# MAGIC   - Click the Experiment icon <img src= "https://docs.databricks.com/en/_images/experiment.png" width=10> in the notebook’s right sidebar
# MAGIC
# MAGIC   - In the Experiment Runs sidebar, click the <img src= "https://docs.databricks.com/en/_images/external-link.png" width=10> icon next to the date of the run. The MLflow Run page displays, showing details of the run, including parameters, metrics, tags, and a list of artifacts.
# MAGIC
# MAGIC   <div style="overflow: hidden; width: 200px; height: 200px;">
# MAGIC     <img src="https://docs.databricks.com/en/_images/quick-start-nb-experiment.png" width=1000">
# MAGIC </div>
# MAGIC
# MAGIC
# MAGIC + **Step 2: Locate Your Experiment:**
# MAGIC
# MAGIC     - Find the experiment name you specified in your MLflow run.
# MAGIC
# MAGIC + **Step 3: Review Run Details:**
# MAGIC
# MAGIC   - Click on the experiment name to view the runs within that experiment.
# MAGIC   - Locate the specific run you want to review.
# MAGIC
# MAGIC + **Step 4: Reviewing Artifacts and Metrics:**
# MAGIC
# MAGIC   - Click on the run to see detailed information.
# MAGIC   - Navigate to the "Artifacts" tab to view logged artifacts.
# MAGIC   - Navigate to the "Metrics" tab to view logged metrics.
# MAGIC
# MAGIC + **Step 5: Viewing Confusion Matrix Image:**
# MAGIC
# MAGIC   - If you logged the confusion matrix as an artifact, you can find it in the "Artifacts" tab.
# MAGIC   - You may find a file named "confusion_matrix.png" (or the specified artifact file name).
# MAGIC   - Download or view the confusion matrix image.
# MAGIC
# MAGIC + **Step 6: View models in the UI:**
# MAGIC   - You can find details about the logged model under the <img src = "https://docs.databricks.com/en/_images/models-icon.png" width = 20> **Models** tab.
# MAGIC   - Look for the model name you specified in your MLflow run (e.g., "decision_tree_model").
# MAGIC
# MAGIC + **Explore Additional Options:**
# MAGIC
# MAGIC   - You can explore other tabs and options in the MLflow UI to gather more insights, such as "Parameters," "Tags," and "Source."
# MAGIC These instructions will guide you through reviewing and exploring the tracked models using the MLflow UI, providing valuable insights into the experiment results and registered models.
