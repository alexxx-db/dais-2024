# Databricks notebook source
# MAGIC %md
# MAGIC # Lab: Instruction Fine-tuning
# MAGIC
# MAGIC This lab demonstrates how to perform instruction fine-tuning (IFT) on a pre-trained language model. 
# MAGIC
# MAGIC Objectives:
# MAGIC
# MAGIC 1. Trigger a single IFT run with specified hyperparameters

# COMMAND ----------

# MAGIC %pip install databricks-genai==1.0.1
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md Set up the classroom to load the variables and datasets needed.

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup

# COMMAND ----------

TABLE = "databricks_blogs_raw"
VOLUME = "blog_ift_data" # Volume containing the instruction dataset

TRAIN_TABLE = "blog_title_generation_train_ift_data"
EVAL_TABLE = "blog_title_generation_eval_ift_data"

UC_MODEL_NAME = "blog_title_generation_llm"  # Name of model registered to Unity Catalog

# COMMAND ----------

# MAGIC %md
# MAGIC ## Question 1: Create a fine-tuning run

# COMMAND ----------

# TODO
# from databricks.model_training import foundation_model as fm

# model = "<FILL_IN>"
# register_to = "<FILL_IN>"
# training_duration = "<FILL_IN>"
# learning_rate = "<FILL_IN>"
# data_prep_cluster_id = spark.conf.get("spark.databricks.clusterUsageTags.clusterId")

# run = fm.create(
#   model=model,
#   train_data_path="<FILL_IN>",
#   eval_data_path="<FILL_IN>",
#   data_prep_cluster_id=data_prep_cluster_id,
#   register_to=register_to,
#   training_duration=training_duration,
#   learning_rate=learning_rate,
# )
# run

# COMMAND ----------

# ANSWER
from databricks.model_training import foundation_model as fm

model =  "meta-llama/Meta-Llama-3-8B-Instruct"
register_to = f"{CATALOG}.{USER_SCHEMA}"
training_duration = "3ep"
learning_rate = "3e-06"
data_prep_cluster_id = spark.conf.get("spark.databricks.clusterUsageTags.clusterId")

run = fm.create(
  model=model,
  train_data_path=f"{CATALOG}.{USER_SCHEMA}.{TRAIN_TABLE}",
  eval_data_path=f"{CATALOG}.{USER_SCHEMA}.{EVAL_TABLE}",
  data_prep_cluster_id=data_prep_cluster_id,
  register_to=register_to,
  training_duration=training_duration,
  learning_rate=learning_rate,
)
run

# COMMAND ----------

run.get_events()
