# Databricks notebook source
# MAGIC %md
# MAGIC # Build All
# MAGIC ## Fine-Tuning Embedding Models and Advanced Retrieval

# COMMAND ----------

# DBTITLE 1,Step 0: Configure build
# MAGIC %md
# MAGIC Check and configure *config.json* for your course build.

# COMMAND ----------

# DBTITLE 1,Step 1: Prepare publisher
import logging
from publisher.workspace_notebook_builder import WorkspaceNotebookBuilder

logging.basicConfig(level=logging.INFO)

builder = WorkspaceNotebookBuilder(
    config_file='config.json',
    api_url=dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().getOrElse(None),
    token=dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)
)

# COMMAND ----------

# DBTITLE 1,Step 2: Generate notebooks
builder.build()

# COMMAND ----------

# DBTITLE 1,Step 3: Generate files for translation
builder.generate_i18n_files()
