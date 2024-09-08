# Databricks notebook source
# INCLUDE_HEADER_TRUE
# INCLUDE_FOOTER_TRUE 

# COMMAND ----------

# MAGIC %md
# MAGIC # Continued Pre-Training and Instruction Fine-tuning
# MAGIC
# MAGIC **Why both?**
# MAGIC
# MAGIC Let's consider a base LLM that is trained on a large corpus of data that is intended to imbue the LLM with a broad and diverse set of knowledge. The goal of pre-training, in the majority of cases, is to teach LLMs how language works. This is done by showing the LLM a vast amount of data that covers all kinds of language. 
# MAGIC
# MAGIC Continued Pre-training (CPT), however, is used to focus an LLM that already has a good grasp of language onto a specific domain of knowledge. The goal of Instruction Fine-tuning (IFT), is to teach the LLM a different way of responding to input. Base and CPT models predict the next token that is most likely to come next from the prompt. IFT models do the same, however they are trained on creating specific responses, rather than continuing on from the prompt text.
# MAGIC
# MAGIC If we want to teach a model a new task, such as creating documentation from a block of code, we need to perform IFT. However, if we can first perform CPT on the base model, we can focus the model first to learn more specifically the domain knowledge, and then when we perform IFT, the model will be better placed to perform well. 
# MAGIC
# MAGIC
# MAGIC ### In this notebook we will:
# MAGIC - Perform CPT on the base model of Llama3-8B
# MAGIC - Use this trained Llama3-8B checkpoint as the starting point for IFT
# MAGIC - Compare this CPT+IFT trained model on the generally IFT trained Llama3-8B model

# COMMAND ----------

# MAGIC %md Set up the classroom to load the variables and datasets needed.

# COMMAND ----------

# MAGIC %pip install databricks-sdk databricks_genai --upgrade
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./Includes/Classroom-Setup

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1. Setting up our CPT data

# COMMAND ----------

# Let's create a new Volume of our own to convert this to .txt format so we can do CPT

# Create a volume to store raw text files for CPT
VOLUME_CPT = "raw_text_cpt"
_ = spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{USER_SCHEMA}.{VOLUME_CPT}")

# COMMAND ----------

pyspark_docs = spark.read.table(f"{CATALOG}.{USER_SCHEMA}.spark_docs_gold")
display(pyspark_docs)

# COMMAND ----------

# MAGIC %md
# MAGIC This takes approximately 6min to run

# COMMAND ----------

from pyspark.sql.functions import udf, pandas_udf, PandasUDFType
from pyspark.sql.types import StringType
from typing import Iterator, List
import pandas as pd
from pyspark.sql import DataFrame
import pyspark.sql.functions as F
from pyspark.sql.types import StringType
import os

# Specify the output directory
output_dir = f"/Volumes/{CATALOG}/{USER_SCHEMA}/{VOLUME_CPT}"

# Define the pandas UDF
@pandas_udf("string", PandasUDFType.SCALAR)
def write_text_file_pandas_udf(id_series: pd.Series, text_series: pd.Series) -> pd.Series:
    file_paths = []
    for id, text in zip(id_series, text_series):
        # Extract the last filename
        filename = id.split("/")[-1]
        # Replace periods with underscores
        modified_filename = filename.replace(".", "_")
        file_path = os.path.join(output_dir, f"{modified_filename}.txt")
        with open(file_path, "w") as file:
            file.write(text)
        file_paths.append(file_path)
    return pd.Series(file_paths)

# Apply the pandas UDF to the DataFrame
df_with_path = pyspark_docs.repartition(200).withColumn("file_path_cpt", write_text_file_pandas_udf("uuid", "content"))

# Use a transformation and action to process all rows
df_with_path.select("file_path_cpt").display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2. Setting up our IFT data

# COMMAND ----------

# We have volume already in ml.genai_professional.finetuning_train_data which has a jsonl for IFT
# Let's create another new Volume of our own to copy this training data over for our IFT 
 
# Create a volume to store raw text files for IFT
VOLUME_IFT = "ift_data"
_ = spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{USER_SCHEMA}.{VOLUME_IFT}")

# COMMAND ----------

@F.udf("string")
def parse_fn_def(fn_string: str) -> str:
    NUMPY_DOCSTRING_START = '"""'
    return fn_string[: fn_string.find(NUMPY_DOCSTRING_START)]

fn_with_docstrings = (
    spark.read.table(f"{CATALOG}.{USER_SCHEMA}.pyspark_code_gold")
    .filter(F.col("doc_string").isNotNull())  # We want only functions with docstrings
    .filter(F.col("element_type") == "function")  # We only want functions, not classes or class methods
    .filter(F.col("args").isNotNull())  # remove null argument functions
    .filter(F.length("args") > 0)  # Remove functions with no arguments
    .filter(F.col("doc_string").contains("Parameters"))  # Get numpy style doc strings
    .withColumn("fn_def", parse_fn_def("source"))
)
display(fn_with_docstrings)

# COMMAND ----------

import json

# Function to format the instructions using a Spark UDF
INSTRUCTION = """The following is a pyspark function definition. Write a numpy style docstring for the provided function definition."""
DEF_KEY = "### Function definition"
RESPONSE_KEY = "### Numpy style docstring"

@F.udf("string")
def format_instructions(fn_def: str) -> str:
    return f"""{INSTRUCTION}\n{DEF_KEY}\n{fn_def}\n{RESPONSE_KEY}\n"""

# Process the DataFrame
ift_df = fn_with_docstrings.select(
    format_instructions(F.col("fn_def")).alias("prompt"),
    F.col("doc_string").alias("response")
)

# Define a UDF to format each row as a message for our Chat Completion model
@F.udf("string")
def format_as_message(prompt, response):
    return json.dumps({
        "messages": [
            {"role": "system", "content": "A conversation between a user and a helpful assistant."},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response}
        ]
    })

# Apply the UDF to the DataFrame
formatted_df = ift_df.withColumn("message", format_as_message(F.col("prompt"), F.col("response")))

# Split the DataFrame into training and evaluation sets
instruction_df_train, instruction_df_eval = formatted_df.randomSplit([0.9, 0.1], seed=42)

# COMMAND ----------

# Function to write DataFrame to JSONL file

def write_df_to_jsonl(df, path):
    temp_path = "/tmp/temp_json"
    df.select("message").coalesce(1).write.mode("overwrite").text(temp_path)

    part_files = dbutils.fs.ls(temp_path)
    json_part_file = next((file.path for file in part_files if file.name.startswith("part-")), None)
    
    if json_part_file:
        final_path = path
        dbutils.fs.mv(json_part_file, final_path)
        dbutils.fs.rm(temp_path, recurse=True)
        print(f"Successfully wrote Spark DataFrame as JSONL to {final_path}")
    else:
        print("No part file found. Check the temp path.")

# Write training and evaluation DataFrames to JSONL files
train_ift_path = f"/Volumes/{CATALOG}/{USER_SCHEMA}/{VOLUME_IFT}/train.jsonl"
eval_ift_path = f"/Volumes/{CATALOG}/{USER_SCHEMA}/{VOLUME_IFT}/eval.jsonl"

write_df_to_jsonl(instruction_df_train, train_ift_path)
write_df_to_jsonl(instruction_df_eval, eval_ift_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3. Fine-tuning API
# MAGIC
# MAGIC #### Databricks Generative AI Fine-tuning API
# MAGIC
# MAGIC In many cases, you may want to use models other than those in the [Foundation API](https://docs.databricks.com/en/machine-learning/foundation-models/index.html). To facilitate this, we will use the new Databricks Mosaic AI Foundation Model Fine-tuning API. This API runs within Databricks and allows you to continuously pre-train a model, as well as instruction finetune it. 
# MAGIC
# MAGIC In this notebook we will cover the important components to configure finetuning runs and work with data in the Unity Catalog. 
# MAGIC
# MAGIC **Prerequisites** 
# MAGIC - For Continued Pre-Training, you will need to have a UC volume with a folder of *.txt files
# MAGIC - For Instruction Fine-Tuning, you will need to have a UC volume with a *.jsonl file that contains `"prompt":"response"` pairs. 
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ```
# MAGIC ft.create(
# MAGIC     model=................. required: which model to use eg. "meta-llama/Meta-Llama-3-8B" 
# MAGIC     train_data_path=....... required: where the data is to train with (must be in the format for either CPT or IFT)    
# MAGIC     eval_data_path=........ required: where the data is to train with (must be in the format for either CPT or IFT)
# MAGIC     register_to=........... required: where to register the model once training has finished eg. <catalog>.<schema>.<custom-name>
# MAGIC     experiment_path=....... optional: a path to store the experiment, otherwise a automated name will be generated based on run name
# MAGIC     task_type=............. optional: INSTRUCTION_FINETUNE (default) or CONTINUED_PRETRAIN or CHAT_COMPLETION
# MAGIC     training_duration=..... optional: defaults to 1ep, can be in # tokens, or epochs eg: 1_000_000tok, 10ep
# MAGIC     custom_weights_path=... optional: this allows us to build from a previously trained model
# MAGIC     )
# MAGIC ```

# COMMAND ----------

from databricks.model_training import foundation_model as ft

# COMMAND ----------

# MAGIC %md
# MAGIC #### Helper Functions 
# MAGIC In addition to the creation of finetuning runs, and monitoring their progress, the Foundation Model API also allows you to:
# MAGIC
# MAGIC - **List** all of the runs in your workspace with `ft.list()` 
# MAGIC - **Cancel** any currently running finetuning run with `ft.cancel()`
# MAGIC - **Delete** any runs from your list with `ft.delete()`
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 3a: Continued Pre-training

# COMMAND ----------

user = DA.catalog_name_prefix
exp_name = "Llama3_CPT"

continued_pretraining_run = ft.create(
    model="meta-llama/Meta-Llama-3-8B" ,
    train_data_path=f"dbfs:/Volumes/{CATALOG}/{USER_SCHEMA}/{VOLUME_CPT}", 
    register_to=f"{CATALOG}.{USER_SCHEMA}",
    experiment_path=f"/Users/{DA.username}/{exp_name}",
    task_type="CONTINUED_PRETRAIN",
    training_duration="10000000tok",
    )
print(f"Finetuning run: {continued_pretraining_run.name} sent to compute cluster")

# COMMAND ----------

# We can track the stages of the run using ft.get_events(fintuning_run.name)
ft.get_events(continued_pretraining_run.name)

# COMMAND ----------

# To see more information of the run we created, we can use the ft.get() command
cpt_info = ft.get(continued_pretraining_run.name)
print(f"Status: {cpt_info.status}\nDetails: {cpt_info.details}")
print(cpt_info)

# COMMAND ----------

# MAGIC %md
# MAGIC **NOTE**:
# MAGIC We will need to wait for this run to complete before using the final checkpoint in the IFT section below

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 3b: Instruction Fine-Tuning (IFT)
# MAGIC
# MAGIC For IFT we can either use a base model, where the weights are loaded in from a public repo, or we can use the checkpoint of the model we pretrained. 

# COMMAND ----------

# MAGIC %md
# MAGIC **Example Training Data Sample**
# MAGIC
# MAGIC Below is an example of the training/evaluation data for the instruction fine-tuning:
# MAGIC ```
# MAGIC {"messages": 
# MAGIC   [
# MAGIC    {"role": "system", "content": "A conversation between a user and a helpful assistant."}, 
# MAGIC    {"role": "user", "content": "The following is a pyspark function definition. Write a numpy style docstring for the provided function definition.\n### Function definition\ndef _update_all_supported_status(\n    all_supported_status: Dict[Tuple[str, str], Dict[str, SupportedStatus]],\n    pd_modules: List[str],\n    pd_module_group: Any,\n    ps_module_group: Any,\n) -> None:\n    \n### Numpy style docstring\n"}, 
# MAGIC    {"role": "assistant", "content": "Update the supported status dictionary with status from multiple modules.\n\nParameters\n----------\nall_supported_status : Dict[Tuple[str, str], Dict[str, SupportedStatus]]\n    The dictionary to update with supported statuses.\npd_modules : List[str]\n    List of module names in pandas.\npd_module_group : Any\n    Importable pandas module group.\nps_module_group : Any\n    Corresponding pyspark.pandas module group."}
# MAGIC   ]
# MAGIC }
# MAGIC
# MAGIC ```

# COMMAND ----------

user = DA.catalog_name_prefix
exp_name = "Llama3_IFT"
custom_weights_path = f"{cpt_info.save_folder}/{cpt_info.name}/checkpoints/latest-sharded-rank0.symlink"
if cpt_info.status.value == "COMPLETED":
    print(f"Checkpoint available at {custom_weights_path}")
else:
    raise Exception(
        f"No model checkpoint available. Please wait for the CPT fine-tuning run to complete before continuing.")

instruction_finetuning_run = ft.create(
    model="meta-llama/Meta-Llama-3-8B" ,
    train_data_path=f"dbfs:/Volumes/{CATALOG}/{USER_SCHEMA}/{VOLUME_IFT}/train.jsonl", 
    eval_data_path=f"dbfs:/Volumes/{CATALOG}/{USER_SCHEMA}/{VOLUME_IFT}/eval.jsonl",
    register_to=f"{CATALOG}.{USER_SCHEMA}", 
    experiment_path=f"/Users/{DA.username}/{exp_name}", 
    task_type="CHAT_COMPLETION",
    training_duration="10000000tok", 
    custom_weights_path=custom_weights_path,
    )

print(f"Finetuning run: {instruction_finetuning_run.name} sent to compute cluster")


# COMMAND ----------

# We can track the stages of the run using ft.get_events(fintuning_run.name)
ft.get_events(instruction_finetuning_run.name)

# COMMAND ----------

# To see more information of the run we created, we can use the ft.get() command
info = ft.get(instruction_finetuning_run.name)
print(f"Status: {info.status}\nDetails: {info.details}")
print(info)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Serving our trained Model

# COMMAND ----------

# MAGIC %md
# MAGIC ![img](https://files.training.databricks.com/images/adv_genai_engineering/M3_1.png)
# MAGIC ![img](https://files.training.databricks.com/images/adv_genai_engineering/M3_2.png)
# MAGIC ![img](https://files.training.databricks.com/images/adv_genai_engineering/M3_3.png)
# MAGIC ![img](https://files.training.databricks.com/images/adv_genai_engineering/M3_4.png)
# MAGIC ![img](https://files.training.databricks.com/images/adv_genai_engineering/M3_5.png)
# MAGIC ![img](https://files.training.databricks.com/images/adv_genai_engineering/M3_6.png)
# MAGIC ![img](https://files.training.databricks.com/images/adv_genai_engineering/M3_7.png)
# MAGIC
# MAGIC
# MAGIC **Note: This will take about 30mins to load**

# COMMAND ----------

# MAGIC %md
# MAGIC You can also use our API

# COMMAND ----------

import mlflow

API_ROOT = mlflow.utils.databricks_utils.get_databricks_host_creds().host
API_TOKEN = mlflow.utils.databricks_utils.get_databricks_host_creds().token

# COMMAND ----------

from mlflow import MlflowClient

mlflow.set_registry_uri("databricks-uc")

model_name = instruction_finetuning_run.name

def get_latest_model_version(model_name):
  client = MlflowClient()
  model_version_infos = client.search_model_versions(f"name = '{model_name}'")
  return max([int(model_version_info.version) for model_version_info in model_version_infos])

latest_model_version = get_latest_model_version(f"{CATALOG}.{USER_SCHEMA}.{model_name}")
print(model_name, latest_model_version)

# COMMAND ----------

import requests

headers = {"Context-Type": "text/json", "Authorization": f"Bearer {API_TOKEN}"}

do_not_update_endpoint = True

if do_not_update_endpoint:
    optimizable_info = requests.get(
        url=f"{API_ROOT}/api/2.0/serving-endpoints/get-model-optimization-info/{CATALOG}.{USER_SCHEMA}.{model_name}/1",
        headers=headers).json()
else:
    optimizable_info = requests.get(
        url=f"{API_ROOT}/api/2.0/serving-endpoints/get-model-optimization-info/{CATALOG}.{USER_SCHEMA}.{model_name}/{latest_model_version}",
        headers=headers).json()
optimizable_info

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ServedEntityInput, EndpointCoreConfigInput

w = WorkspaceClient()

endpoint_name = f"adv_genai_cpt_ift_model"

try:
    endpoint = w.serving_endpoints.get(endpoint_name)
    endpoint_exists = True
except:
    endpoint_exists = False

print(f"Endpoint exists: {endpoint_exists}")

if endpoint_exists and do_not_update_endpoint:
    print("Reusing existing endpoint...")
elif endpoint_exists and not do_not_update_endpoint:
    try:
        print(f"Updating endpoint to model version {latest_model_version}")
        print(w.serving_endpoints.update_config_and_wait(
            name=endpoint_name, 
            served_entities=[
                ServedEntityInput(
                    entity_name=f"{CATALOG}.{USER_SCHEMA}.{model_name}",
                    entity_version=str(latest_model_version),
                    max_provisioned_throughput=optimizable_info["throughput_chunk_size"], 
                    min_provisioned_throughput=0,
                    scale_to_zero_enabled=True
            )]
        ))
        print("Updating...")
    except:
        print("Update failed. Check your permissions and other ongoing tasks.")
else:
    try:
        print("Creating new serving endpoint...")
        print(w.serving_endpoints.create_and_wait(
            name=endpoint_name,
            config=EndpointCoreConfigInput(
                name=endpoint_name,
                served_entities=[
                    ServedEntityInput(
                        entity_name=f"{CATALOG}.{USER_SCHEMA}.{model_name}",
                        entity_version=str(latest_model_version),
                        max_provisioned_throughput=optimizable_info["throughput_chunk_size"], 
                        min_provisioned_throughput=0,
                        scale_to_zero_enabled=True
                    ),
                ]
            )
        ))
    except Exception as e:
        print(e)
        print("Creation failed. Check your permissions and other ongoing tasks")


# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Model Inference
# MAGIC
# MAGIC We can compare the CPT+IFT model that we've built, to the standard IFT model on the Model Serving Dashboard using the databricks inference client

# COMMAND ----------

prompt = "The following is a pyspark function definition. Write a numpy style docstring for the provided function definition.\n### Function Definition\ndef _update_all_supported_status(\n    all_supported_status: Dict[Tuple[str, str], Dict[str, SupportedStatus]],\n    pd_modules: List[str],\n    pd_module_group: Any,\n    ps_module_group: Any,\n) -> None:\n    \n### Numpy style docstring\n"

temperature = 1.0 
max_tokens = 256

# COMMAND ----------

# MAGIC %md
# MAGIC Send the query to our fine-tuned model endpoint

# COMMAND ----------

from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

w_response_ft = w.serving_endpoints.query(name=endpoint_name, 
                                          messages=[ChatMessage(role=ChatMessageRole.USER, content=prompt)], 
                                          temperature=temperature, 
                                          max_tokens=max_tokens)
w_response_ft.choices[0].message.content

# COMMAND ----------

# MAGIC %md
# MAGIC Compare the fine-tuned response with the foundation model's response.

# COMMAND ----------

w_response_fm_api = w.serving_endpoints.query(name="databricks-meta-llama-3-1-70b-instruct", 
                                              messages=[ChatMessage(role=ChatMessageRole.USER, content=prompt)],
                                              temperature=temperature, 
                                              max_tokens=max_tokens)
w_response_fm_api.choices[0].message.content

# COMMAND ----------


