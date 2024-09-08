# Databricks notebook source


# COMMAND ----------

# MAGIC %pip install "mlflow-skinny[databricks]>=2.4.1"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import mlflow
# alex_barreto_xt2o_da.default.ml_model
catalog = "alex_barreto_xt2o_da"
schema = "default"
model_name = "ml_model"
mlflow.set_registry_uri("databricks-uc")
mlflow.register_model("runs:/88651e2488454a5fb0ce0412e61b3d5f/decision_tree", f"{catalog}.{schema}.{model_name}")
