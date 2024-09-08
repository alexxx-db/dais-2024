# Databricks notebook source
# INCLUDE_HEADER_FALSE
# INCLUDE_FOOTER_FALSE

# COMMAND ----------

# WORKSPACE_IDENTIFIER = spark.conf.get("spark.databricks.workspaceUrl").split(".")[0].replace("-", "")
# CATALOG = f"dbacademy_adv_genai_{WORKSPACE_IDENTIFIER}" # catalog shared across all workspace students
# USER_SCHEMA = spark.sql("SELECT USER()").first()[0].split("@")[0].replace(".", "_")
# SHARED_SCHEMA = "shared_adv_genai_schema"

# # Catalog creation 
# _ = spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
# _ = spark.sql(f"USE CATALOG {CATALOG}")

# # Schema creation
# _ = spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{USER_SCHEMA}")
# _ = spark.sql(f"USE SCHEMA {USER_SCHEMA}")
# _ = spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SHARED_SCHEMA}")

# datasets_path = DA.paths.datasets.replace("/dbfs", "dbfs:")

# print(f"This lesson will use {CATALOG}.{USER_SCHEMA}. Use variables `CATALOG` and `USER_SCHEMA` as needed for a user-specific schema to write your data to. If you're running this outside of the classroom environment, replace the variables `CATALOG` and `USER_SCHEMA` with your own locations (note that this must be Unity Catalog and not the hive metastore).")

# COMMAND ----------

# CATALOG = spark.sql("SELECT current_catalog()").first()[0]
WORKSPACE_IDENTIFIER = spark.conf.get("spark.databricks.workspaceUrl").split(".")[0].replace("-", "")
CATALOG = f"dbacademy_adv_genai_{WORKSPACE_IDENTIFIER}" # catalog shared across all workspace students
USER_SCHEMA = spark.sql("SELECT USER()").first()[0].split("@")[0].replace(".", "_")
# SHARED_SCHEMA = "shared_adv_genai_schema"

CATALOG = "users"
SHARED_CATALOG = CATALOG
SHARED_SCHEMA = USER_SCHEMA

# Catalog creation
try:
    _ = spark.sql(f"USE CATALOG {CATALOG}")
except:
    _ = spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")

# Schema creation
try:
    _ = spark.sql(f"USE SCHEMA {CATALOG}.{USER_SCHEMA}")
except:
    _ = spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{USER_SCHEMA}")

try:
    _ = spark.sql(f"USE SCHEMA {SHARED_CATALOG}.{SHARED_SCHEMA}")
except:
    _ = spark.sql(f"CREATE SCHEMA IF NOT EXISTS {SHARED_CATALOG}.{SHARED_SCHEMA}")

datasets_path = DA.paths.datasets.replace("/dbfs", "dbfs:")

print(f"This lesson will use {CATALOG}.{USER_SCHEMA}. Use variables `CATALOG` and `USER_SCHEMA` as needed for a user-specific schema to write your data to. If you're running this outside of the classroom environment, replace the variables `CATALOG` and `USER_SCHEMA` with your own locations (note that this must be Unity Catalog and not the hive metastore).")

# COMMAND ----------

def create_tables(table_name: str, relative_path: str, schema: str, datasets_path: str = datasets_path, catalog : str = CATALOG) -> None:
    """
    Create a Delta table from a Delta file at the specified path.

    Parameters:
    - table_name (str): The name of the table to be created.
    - relative_path (str): The relative path to the Delta file.
    - datasets_path (str): The base path where datasets are stored.
    - schema (str): The schema to use for the table.

    Returns:
    - None
    """
    path = f"{datasets_path}/{relative_path}"
    table_name = f"{catalog}.{schema}.{table_name}"

    df = spark.read.format("delta").load(path)
    spark.sql(f"DROP TABLE IF EXISTS {table_name}")
    df.write.saveAsTable(table_name)

    print(f"Created table {table_name}")


# Uncomment tables as needed

create_tables("blogs_bronze", "blogs/bronze", USER_SCHEMA)

# create_tables("pyspark_code_bronze", "pyspark-code/bronze", USER_SCHEMA)
create_tables("pyspark_code_gold", "pyspark-code/gold", USER_SCHEMA) 
# create_tables("pyspark_code_gold_flat", "pyspark-code/gold-flat", USER_SCHEMA) 
# create_tables("llm_output_df", "pyspark-code/llm_output_df", USER_SCHEMA) 

# create_tables("generated_questions_eval", "spark-docs/generated_questions_eval", USER_SCHEMA)
# create_tables("generated_questions_train", "spark-docs/generated_questions_train", USER_SCHEMA)
create_tables("spark_docs_gold", "spark-docs/gold", USER_SCHEMA)
# create_tables("spark_docs_gold_flat", "spark-docs/gold-flat", USER_SCHEMA)
# _ = spark.sql(f"ALTER TABLE {CATALOG}.{USER_SCHEMA}.spark_docs_gold_flat SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
# create_tables("spark_docs_bronze", "spark-docs/txt-raw", USER_SCHEMA)

print()
print("Completed table creation")


