# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning">
# MAGIC </div>
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC # Fine-Tuning an Embedding Model
# MAGIC
# MAGIC It's often desirable to fine-tune your out-of-the-box embedding model to see better retrieval performance on your domain-specific data. 
# MAGIC
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) In this lesson you:<br>
# MAGIC  - Prepare necessary training and evaluation data for fine-tuning embedding model 
# MAGIC  - Fine-tune a popular embedding model 
# MAGIC  - Embed the dataset to Vector Search index using the newly fine-tuned embedding model 
# MAGIC  - Serve the model using Provisioned Throughput 
# MAGIC  - Evaluate the model using recall metrics and manual inspection
# MAGIC
# MAGIC In this notebook, we will fine-tune our out-of-the-box embedding model based on PySpark documentation. We will then compare retrieval performances when using a generic embedding model versus a fine-tuned embedding model.

# COMMAND ----------

# MAGIC %pip install mlflow==2.14.1 sentence-transformers==3.0.1 torch==2.3.0 transformers==4.40.1 databricks-sdk==0.31.1 tf-keras==2.16.0 accelerate==0.27.2
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./Includes/Classroom-Setup

# COMMAND ----------

import pandas as pd
import matplotlib.pyplot as plt
import pyspark.sql.functions as F
from random import randint
from typing import List, Callable

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors.platform import ResourceDoesNotExist, ResourceAlreadyExists
from databricks.sdk.service.vectorsearch import VectorIndexType, DeltaSyncVectorIndexSpecResponse, EmbeddingSourceColumn, PipelineType
from databricks.sdk.service.serving import ServedEntityInput, EndpointCoreConfigInput

from langchain_core.embeddings import Embeddings

from mlflow.tracking.client import MlflowClient

from datasets import Dataset 

import sentence_transformers, requests, time, json, mlflow, yaml, os, torch
from sentence_transformers import InputExample, losses
from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
from sentence_transformers import SentenceTransformer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, SimilarityFunction
from transformers import AutoTokenizer

from torch.utils.data import DataLoader
import torch

import tempfile

from databricks.sdk.service.serving import AutoCaptureConfigInput

# COMMAND ----------

# MAGIC %sh PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True # Prevent PyTorch from hogging too much GPU memory in reserve.

# COMMAND ----------

VS_ENDPOINT = "adv_genai_course_endpoint_1" 
VS_INDEX = "spark_docs_bge_finetuned"

DBRX_ENDPOINT = "databricks-dbrx-instruct"
LLAMA3_ENDPOINT = "databricks-meta-llama-3-1-70b-instruct"
GENERATED_QUESTIONS = "generated_questions"
GOLD_CHUNKS_FLAT = "spark_docs_gold_flat"

gpu_is_available = torch.cuda.is_available()

# COMMAND ----------

VS_INDEX_FULL_NAME = f"{SHARED_CATALOG}.{SHARED_SCHEMA}.{VS_INDEX}"

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC We load an open-sourced embedding model. Then, we will fine-tune this embedding model based on the training data.

# COMMAND ----------

# Load a desired model. If GPU is unavailable, load a smaller variant.
model = SentenceTransformer("BAAI/bge-large-en") if gpu_is_available else SentenceTransformer("BAAI/bge-small-en")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Before we can fine-tune the embedding model, we need to make sure that our training data conforms to the format that the `sentence transformer` library expects. From the `sentence transformer` library, we can represent our training data using the `InputExample` class. Refer to [documentation here](https://www.sbert.net/docs/training/overview.html#training-data).
# MAGIC
# MAGIC >As parameters, it accepts texts, which is a list of strings representing our pairs (or triplets). Further, we can also pass a label (either float or int).

# COMMAND ----------

training_set = spark.table(f"{CATALOG}.{USER_SCHEMA}.{GENERATED_QUESTIONS}_train").toPandas()
eval_set = spark.table(f"{CATALOG}.{USER_SCHEMA}.{GENERATED_QUESTIONS}_eval").toPandas()
display(training_set)

# COMMAND ----------

for index, row in training_set.head(1).iterrows():
    print(f"index = {index}, question = {row.generated_question}, context = {row.content}")

# COMMAND ----------

# MAGIC %md
# MAGIC Recall, the example from the slides:
# MAGIC
# MAGIC >Triplets consist of an anchor, positive example, and negative example.
# MAGIC
# MAGIC - Anchor: When was the first Land Rover designed?
# MAGIC - Positive: “The design for the original vehicle was started in 1947 by Maurice Wilks. Wilks, chief designer at the Rover Company, on his farm [...]”
# MAGIC - Negative: “Land Rover is a British brand of predominantly four-wheel drive, off-road capable vehicles, owned by multinational car manufacturer Jaguar Land Rover (JLR)”
# MAGIC
# MAGIC The `MultipleNegativesRankingLoss` and `CachedMultipleNegativesRankingLoss` only need to be supplied with anchor-positive pairs, and then it will create anchor-positive-negative triplets by using all other in-batch positive examples as negatives for each supplied anchor-positive.

# COMMAND ----------

column_remap = {"content" : "anchor",  "generated_question" : "positive"}

ft_train_dataset = Dataset.from_pandas(
    training_set.rename(columns=column_remap)
)

ft_eval_dataset = Dataset.from_pandas(
    eval_set.rename(columns=column_remap)
)

# COMMAND ----------

# MAGIC %md
# MAGIC # Step 3: Define loss function

# COMMAND ----------

train_loss = CachedMultipleNegativesRankingLoss(
    model=model, 
    mini_batch_size=8
)

# COMMAND ----------

# MAGIC %md
# MAGIC # Step 4: Specify Training Arguments (Optional)

# COMMAND ----------

num_epochs = 1

temp_checkpoint_dir = tempfile.TemporaryDirectory().name
dbutils.fs.mkdirs(temp_checkpoint_dir)

args = SentenceTransformerTrainingArguments(
    # Required parameters:
    output_dir=temp_checkpoint_dir, # Specify where outputs go
    # Optional training parameters:
    num_train_epochs=num_epochs, # How many full passes over the data should be done during training (epochs)?
    learning_rate=2e-5, # This takes trial and error, but 2e-5 is a good starting point.
    auto_find_batch_size=True, # Allow automatic determination of batch size
    warmup_ratio=1, # This takes trial and error
    seed=42, # Seed for reproducibility
    data_seed=42, # Seed for reproducibility
    # Optional tracking/debugging parameters:
    evaluation_strategy="steps",
    eval_steps=10, # How often evaluation loss should be logged
    logging_steps=10 # How often training loss should be calculated
)

# COMMAND ----------

# MAGIC %md
# MAGIC # Step 5: Fine-tune embedding model
# MAGIC
# MAGIC Here, we define training loss to be Multiple Negatives Ranking Loss (MNRL). MNRL is useful for constrastive learning, where we identify similar vs. dissimilar pairs of examples. Refer to [docs here](https://www.sbert.net/docs/package_reference/losses.html#sentence_transformers.losses.CachedMultipleNegativesRankingLoss). 
# MAGIC

# COMMAND ----------

trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=ft_train_dataset,
    eval_dataset=ft_eval_dataset,
    loss=train_loss
)

with mlflow.start_run() as run:
    trainer.train()

# COMMAND ----------

# MAGIC %md
# MAGIC # Serve the embedding model with Provisioned Throughput
# MAGIC
# MAGIC ## Log model 
# MAGIC Our fine-tuned `sentence-transformer` is ready. If we want to serve it with provisioned throughput we must log it to mlflow as a `transformer` flavor model with extra `metadata={"task": "llm/v1/embeddings", "model_type": "bge-small"}`

# COMMAND ----------

registered_ft_embedding_model_name = "bge_finetuned"
data = "Look at my finetuned model"

# Log the model to unity catalog
mlflow.set_registry_uri("databricks-uc")

signature = mlflow.models.infer_signature(
    model_input=data,
    model_output=model.encode(data),
)

with mlflow.start_run() as run:
  # Extract the transformer and tokenizer components from the sentence-transformer
  components = {
      "model": model[0].auto_model,
      "tokenizer": model[0].tokenizer,
  }

  # log the model to mlflow as a transformer with PT metadata
  _logged = mlflow.transformers.log_model(
      transformers_model=components,
      artifact_path="model",
      task="llm/v1/embeddings",
      registered_model_name=f"{CATALOG}.{USER_SCHEMA}.{registered_ft_embedding_model_name}",
      metadata={
        "model_type": "bge-large" if gpu_is_available else "bge-small" # Can be bge-small or bge-large
        },
      input_example={"input": data}
  )

# COMMAND ----------

def get_latest_model_version(model_name):
  client = MlflowClient()
  model_version_infos = client.search_model_versions(f"name = '{model_name}'")
  return max([int(model_version_info.version) for model_version_info in model_version_infos])

# If instructor needs to update the model, the schema needs to change to SHARED_SCHEMA
latest_model_version = get_latest_model_version(f"{CATALOG}.{USER_SCHEMA}.{registered_ft_embedding_model_name}")
latest_model_version

# COMMAND ----------

# Get the API endpoint and token for the current notebook context
API_ROOT = mlflow.utils.databricks_utils.get_databricks_host_creds().host
API_TOKEN = mlflow.utils.databricks_utils.get_databricks_host_creds().token

headers = {"Context-Type": "text/json", "Authorization": f"Bearer {API_TOKEN}"}

do_not_update_endpoint = True
mlflow.set_registry_uri("databricks-uc")

if do_not_update_endpoint:
    optimizable_info = requests.get(
        url=f"{API_ROOT}/api/2.0/serving-endpoints/get-model-optimization-info/{SHARED_CATALOG}.{SHARED_SCHEMA}.{registered_ft_embedding_model_name}/{latest_model_version}",
        headers=headers).json()
else:
    optimizable_info = requests.get(
        url=f"{API_ROOT}/api/2.0/serving-endpoints/get-model-optimization-info/{SHARED_CATALOG}.{SHARED_SCHEMA}.{registered_ft_embedding_model_name}/{latest_model_version}",
        headers=headers).json()


optimizable_info

# COMMAND ----------

# MAGIC %md
# MAGIC The endpoint spin-up time could take ~15 mins

# COMMAND ----------

w = WorkspaceClient()

endpoint_name = f"adv_genai_{registered_ft_embedding_model_name}"

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
                    entity_name=f"{SHARED_CATALOG}.{SHARED_SCHEMA}.{registered_ft_embedding_model_name}",
                    entity_version=str(latest_model_version), 
                    max_provisioned_throughput=optimizable_info["throughput_chunk_size"], 
                    min_provisioned_throughput=optimizable_info["throughput_chunk_size"]
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
                        entity_name=f"{SHARED_CATALOG}.{SHARED_SCHEMA}.{registered_ft_embedding_model_name}",
                        entity_version=str(latest_model_version),
                        max_provisioned_throughput=optimizable_info["throughput_chunk_size"], 
                        min_provisioned_throughput=optimizable_info["throughput_chunk_size"]
                    ),
                ]
            )
        ))
    except:
        print("Creation failed. Check your permissions and other ongoing tasks")

# COMMAND ----------

# MAGIC %md
# MAGIC # Set up Vector Search index

# COMMAND ----------

try:
    index = w.vector_search_indexes.get_index(f"{SHARED_CATALOG}.{SHARED_SCHEMA}.{VS_INDEX}")
    index_exists = True
    print("VS index already exists")
except:
    print(f"VS Index, {VS_INDEX}, did not already exist.")
    index_exists = False

if index_exists:
    print("Syncing existing index")
    w.vector_search_indexes.sync_index(f"{SHARED_CATALOG}.{SHARED_SCHEMA}.{VS_INDEX}") # If it exists, sync
else:
    print(f"Creating vector index: {CATALOG}.{USER_SCHEMA}.{VS_INDEX}")
    source_table = f"{CATALOG}.{USER_SCHEMA}.{GOLD_CHUNKS_FLAT}"
    _ = spark.sql(f"ALTER TABLE {source_table}  SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
    w.vector_search_indexes.create_index(
        name=f"{CATALOG}.{USER_SCHEMA}.{VS_INDEX}",
        endpoint_name=VS_ENDPOINT,
        primary_key="uuid",
        index_type=VectorIndexType("DELTA_SYNC"),
        delta_sync_index_spec=DeltaSyncVectorIndexSpecResponse(
            embedding_source_columns=[
                EmbeddingSourceColumn(
                    name="content",
                    embedding_model_endpoint_name=endpoint_name
                )],
            pipeline_type=PipelineType("TRIGGERED"),
            source_table=source_table
                   )
    )

# COMMAND ----------

# MAGIC %md
# MAGIC The following cell can take ~7 mins.

# COMMAND ----------

while w.vector_search_indexes.get_index(f"{SHARED_CATALOG}.{SHARED_SCHEMA}.{VS_INDEX}").status.ready == False:
    print(f"Waiting for vector search creation or sync to complete for {SHARED_CATALOG}.{SHARED_SCHEMA}.{VS_INDEX}")
    time.sleep(30) # Give it some time to finish

print(w.vector_search_indexes.get_index(f"{SHARED_CATALOG}.{SHARED_SCHEMA}.{VS_INDEX}").status)

# COMMAND ----------

# MAGIC %md
# MAGIC # Evaluate fine-tuned embedding model 
# MAGIC Wait for VS index to be created and then perform eval

# COMMAND ----------

def get_relevant_documents(question : str, index_name : str, k : int = 3, filters : str = None, max_retries : int = 3) -> List[dict]:
    response_received = False
    retries = 0
    while ((response_received == False) and (retries < max_retries)):
        try:
            docs = w.vector_search_indexes.query_index(
                index_name=index_name,
                columns=["uuid","content","category","filepath"],
                filters_json=filters,
                num_results=k,
                query_text=question
            )
            response_received = True
            docs_pd = pd.DataFrame(docs.result.data_array)
            docs_pd.columns = [_c.name for _c in docs.manifest.columns]
        except Exception as e:
            retries += 1
            time.sleep(1 * retries)
            print(e)
    return json.loads(docs_pd.to_json(orient="records"))

# COMMAND ----------

index_full_new = f"{SHARED_CATALOG}.{SHARED_SCHEMA}.{VS_INDEX}"
index_full_old = f"{SHARED_CATALOG}.{SHARED_SCHEMA}.spark_docs_gold_index"

def get_relevant_doc_ids(question : str, index_name : str) -> list[str]:
    docs = get_relevant_documents(question, index_name=index_name, k=10)
    return [_x["uuid"] for _x in docs]

# COMMAND ----------

eval_df = spark.table(f"{CATALOG}.{USER_SCHEMA}.{GENERATED_QUESTIONS}_eval")

eval_pd_new = eval_df.toPandas()
eval_pd_new["uuid"] = eval_pd_new["uuid"].transform(lambda x: [x])
eval_pd_new["retrieved_docs"] = eval_pd_new["generated_question"].transform(lambda x: get_relevant_doc_ids(x, index_full_new))

eval_pd_old = eval_df.toPandas()
eval_pd_old["uuid"] = eval_pd_old["uuid"].transform(lambda x: [x])
eval_pd_old["retrieved_docs"] = eval_pd_old["generated_question"].transform(lambda x: get_relevant_doc_ids(x, index_full_old))

display(eval_pd_new)

# COMMAND ----------

with mlflow.start_run() as run:
    eval_results_ft = mlflow.evaluate(
        data=eval_pd_new,
        model_type="retriever",
        targets="uuid",
        predictions="retrieved_docs",
        evaluators="default",
        extra_metrics=[mlflow.metrics.recall_at_k(i) for i in range(1,10,1)]
    )
    eval_results = mlflow.evaluate(
        data=eval_pd_old,
        model_type="retriever",
        targets="uuid",
        predictions="retrieved_docs",
        evaluators="default",
        extra_metrics=[mlflow.metrics.recall_at_k(i) for i in range(1,10,1)]
    )

# COMMAND ----------

plt.plot([eval_results_ft.metrics[f"recall_at_{i}/mean"] for i in range(1,10,1)], label="finetuned")
plt.plot([eval_results.metrics[f"recall_at_{i}/mean"] for i in range(1,10,1)], label="bge")
plt.title("Recall at k")
plt.xlabel("k")
plt.legend()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Based on the plot above, we see that the fine-tuned model performs better overall!

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Compare answers retrieving from documents indexed using non-fine-tuned vs fine-tuned embedding model

# COMMAND ----------

example_question = "What function should if I have an array column and I want to make each item in the array a separate row?"

# COMMAND ----------

get_relevant_documents(example_question, index_full_old, k=2)

# COMMAND ----------

get_relevant_documents(example_question, index_full_new, k=2)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC &copy; 2024 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the 
# MAGIC <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/><a href="https://databricks.com/privacy-policy">Privacy Policy</a> | 
# MAGIC <a href="https://databricks.com/terms-of-use">Terms of Use</a> | 
# MAGIC <a href="https://help.databricks.com/">Support</a>
