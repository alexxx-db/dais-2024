# Databricks notebook source
# MAGIC %pip install "mlflow-skinny[databricks]>=2.4.1"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import mlflow
catalog = "alex_barreto_xt2o_da"
schema = "default"
model_name = "custom_ml_model"
mlflow.set_registry_uri("databricks-uc")
mlflow.register_model("runs:/e6e2d023ded543c09d1d5ecbbcf34d25/model", f"{catalog}.{schema}.{model_name}")
