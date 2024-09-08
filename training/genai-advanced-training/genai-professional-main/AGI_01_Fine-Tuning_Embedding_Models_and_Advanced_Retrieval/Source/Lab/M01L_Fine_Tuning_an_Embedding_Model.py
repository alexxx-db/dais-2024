# Databricks notebook source
# INCLUDE_HEADER_TRUE
# INCLUDE_FOOTER_TRUE

# COMMAND ----------

# MAGIC %md
# MAGIC #Lab: Fine-Tuning an Embedding Model
# MAGIC
# MAGIC Now that you've seen how much fine-tuning the embedding model can improve its performance, try so for yourself:
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) In this lab you will:<br>
# MAGIC  - Prepare necessary training and evaluation data for fine-tuning embedding model 
# MAGIC  - Fine-tune a small embedding model 
# MAGIC  - Embed the dataset to Vector Search index using the newly fine-tuned embedding model 
# MAGIC  - Evaluate the model's improvement
# MAGIC
# MAGIC As a user, you would like to ask questions about past Databricks blog posts. Since Databricks blog posts can contain lots of data engineering, machine learning or Databrick-specific terminology, you would like the embedding model to learn from these jargons. 

# COMMAND ----------

# MAGIC %pip install --quiet databricks-sdk==0.24.0 mlflow==2.14.1 unstructured==0.13.7 sentence-transformers==3.0.1 torch==2.3.0 transformers==4.40.1 accelerate==0.27.2

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup

# COMMAND ----------


import json, sentence_transformers, torch, yaml, os, gc, logging, time, requests, mlflow
import matplotlib.pyplot as plt
import pandas as pd
import pyspark.sql.functions as F

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors.platform import NotFound, ResourceDoesNotExist
from databricks.sdk.service.vectorsearch import EndpointType, VectorIndexType, DeltaSyncVectorIndexSpecResponse, EmbeddingSourceColumn, PipelineType
from databricks.sdk.service.serving import ServedEntityInput, EndpointCoreConfigInput, ChatMessage, ChatMessageRole
from langchain_core.embeddings import Embeddings
from mlflow.tracking.client import MlflowClient
from pyspark.sql.types import StringType, StructField, StructType, ArrayType, IntegerType
from random import randint
from sentence_transformers import InputExample, losses, SentenceTransformer, SentencesDataset
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, TripletEvaluator

from sentence_transformers.training_args import SentenceTransformerTrainingArguments 
from sentence_transformers.trainer import SentenceTransformerTrainer

from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from typing import List, Callable
from unstructured.chunking.basic import chunk_elements
from unstructured.partition.text import partition_text
import tempfile

# COMMAND ----------

# MAGIC %md
# MAGIC *Note*: Must raise permissions for the newly created for the FT endpoint from CAN_QUERY to CAN_MANAGE for all users. This embedding model is used for the vector search index 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Create Data for RAG

# COMMAND ----------

# Let's have a look at the dataset we are using for the blogs data
display(spark.read.table(f"{CATALOG}.{USER_SCHEMA}.blogs_bronze"))

# COMMAND ----------

# MAGIC %md
# MAGIC # Step 1: Chunk the data and save to vector search index

# COMMAND ----------

# # TODO
# Now create a function to get the blog chunks so we can create our vector index
# @F.udf(ArrayType(StructType([
#   StructField("content", StringType(), True),
#   StructField("category", StringType(), True),
#   StructField("char_length", IntegerType(), True),
#   StructField("chunk_num", IntegerType(), True)
# ])))

def get_blog_chunks(blog_text : str, max_characters : int, new_after_n_chars : int) -> list:
  elements = partition_text(text=blog_text)
  chunks = chunk_elements(elements, max_characters=max_characters, new_after_n_chars=new_after_n_chars)
  ret = []
#   for i,_chunk in enumerate(<FILL_IN>):
      # <FILL_IN> 
#   return ret

# COMMAND ----------

# ANSWER

#Assuming the blog data is in raw form in the delta table, we can do some quick chunking 

@F.udf(ArrayType(StructType([
  StructField("content", StringType(), True),
  StructField("category", StringType(), True),
  StructField("char_length", IntegerType(), True),
  StructField("chunk_num", IntegerType(), True)
])))

def get_blog_chunks(blog_text : str, max_characters : int, new_after_n_chars : int) -> list:
  elements = partition_text(text=blog_text)
  chunks = chunk_elements(elements, max_characters=max_characters, new_after_n_chars=new_after_n_chars)
  ret = []
  for i,_chunk in enumerate(chunks):
    _cat = _chunk.category
    _content = _chunk.metadata.text_as_html if (_cat == "Table") else _chunk.text # Keep HTML for tables to improve retrieval of table values.
    ret.append({"content": _content, "category": _cat, "char_length": len(_content), "chunk_num": i})
  return ret

# COMMAND ----------

# # TODO
# Using the function we defined above, create the chunks and save them as a new column in the bronze table

# blog_chunks = "<FILL_IN>"
# blog_chunks.limit(2).display()

# COMMAND ----------

# ANSWER

blog_chunks = (
  spark
  .table(f"{CATALOG}.{USER_SCHEMA}.blogs_bronze")
  .withColumn(
    "chunks",
    get_blog_chunks(F.column("text"), F.lit(1500), F.lit(500))
  )
  .select(F.col("url"),F.explode("chunks").alias("chunk"))
.select(F.col("url"),"chunk.*")
.withColumn("uuid",F.concat_ws("_","url","chunk_num"))
)
display(blog_chunks.limit(2))

# COMMAND ----------

# Now we save the gold table and setup a vector search endpoint 
(
  blog_chunks
  .write
  .option("delta.enableChangeDataFeed", "true") # enable CDF
  .mode("overwrite")
  .saveAsTable(f"{CATALOG}.{USER_SCHEMA}.gold_blog_chunks") 
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Create the Vector Search Index

# COMMAND ----------

# MAGIC %md First, confirm the pre-created endpoint to use is available. If you're running this in your own env, you can use `w.vector_search_endpoints.create_endpoint_and_wait(<FILL_IN>)` to generate the endpoint. 

# COMMAND ----------

# VS_ENDPOINT = f"vs_endpoint_{(sum(ord(char) for char in DA.unique_name('_')) % 9) + 1}"
VS_ENDPOINT = "adv_genai_course_endpoint_1"

w = WorkspaceClient()

try:
    endpoint = w.vector_search_endpoints.get_endpoint(VS_ENDPOINT)
    print(f"Endpoint {VS_ENDPOINT} found. Using this endpoint for your index.")
except:
    print(f"Endpoint {VS_ENDPOINT} not found. Please confirm the endpoint has been set up.")
    assert(False)

# COMMAND ----------

# MAGIC %md Define other constants.

# COMMAND ----------

VS_INDEX = f"db_blog_{USER_SCHEMA}" # Creating a personalized index in your endpoint

DBRX_ENDPOINT = "databricks-dbrx-instruct"
GENERATED_QUESTIONS = "generated_questions"
GOLD_CHUNKS_FLAT = "gold_chunks_flat"

# COMMAND ----------

# # TODO
# Now that there is an endpoint to host the index, create the index using the table with the chunks from above

# try:
    # w.vector_search_indexes.sync_index("<FILL_IN>")
# except ResourceDoesNotExist as e:
    # w.vector_search_indexes.create_index("<FILL_IN>")

# COMMAND ----------

# ANSWER

try:
    w.vector_search_indexes.sync_index(f"{CATALOG}.{USER_SCHEMA}.{VS_INDEX}")
except ResourceDoesNotExist as e:
    w.vector_search_indexes.create_index(
        name=f"{CATALOG}.{USER_SCHEMA}.{VS_INDEX}",
        endpoint_name=VS_ENDPOINT,
        primary_key="uuid",
        index_type=VectorIndexType("DELTA_SYNC"),
        delta_sync_index_spec=DeltaSyncVectorIndexSpecResponse(
            embedding_source_columns=[
                EmbeddingSourceColumn(
                    name="content",
                    embedding_model_endpoint_name="databricks-bge-large-en"
                )],
            pipeline_type=PipelineType("TRIGGERED"),
            source_table=f"{CATALOG}.{USER_SCHEMA}.gold_blog_chunks"
        )
    )

# COMMAND ----------

# MAGIC %md Check to see if the index is ready (this could take a few minutes).

# COMMAND ----------

status = w.vector_search_indexes.get_index(f"{CATALOG}.{USER_SCHEMA}.{VS_INDEX}").as_dict()

print(status.get("status")["ready"])

status

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Building The Retriever

# COMMAND ----------

# # TODO:
# Complete the function below that will return relevant documents from the vector search index
# def get_relevant_documents(question:str, index_name:str, k:int = 3, filters:str = None, max_retries:int = 3) -> List[dict]:
#     "<FILL_IN>"


# COMMAND ----------

# ANSWER

def get_relevant_documents(question:str, index_name:str, k:int = 3, filters:str = None, max_retries:int = 3) -> List[dict]:
    """
    This function searches through the supplied vector index name and returns relevant documents 
    """
    docs = w.vector_search_indexes.query_index(
        index_name=index_name,
        columns=["uuid", "content", "category", "url"],
        filters_json=filters,
        num_results=k,
        query_text=question
    )
    docs_pd = pd.DataFrame(docs.result.data_array)
    docs_pd.columns = [_c.name for _c in docs.manifest.columns]
    return json.loads(docs_pd.to_json(orient="records"))

# COMMAND ----------

# NOTE: This can take up to 7min to be ready
while w.vector_search_indexes.get_index(f"{CATALOG}.{USER_SCHEMA}.{VS_INDEX}").status.ready is not True:
    print("Vector search index is not ready yet...")
    time.sleep(30)

print("Vector search index is ready")

# COMMAND ----------

# # TODO: 
# VS_INDEX_FULL_NAME="<FILL_IN>"
# get_relevant_documents("<FILL_IN>")

# COMMAND ----------

# ANSWER

VS_INDEX_FULL_NAME=f"{CATALOG}.{USER_SCHEMA}.{VS_INDEX}"
get_relevant_documents("What is the Chronobase?", VS_INDEX_FULL_NAME, k=10)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Generating Synthetic Questions from Raw Data

# COMMAND ----------

# # TODO:

# Using the prompts below, create the llm requests and send them to the serving endpoint.

# # Generate Questions 
# SYSTEM_PROMPT = """You are a journalist who is going to ask a Databricks usage question related to the provided context. Answer only with your question related to the provided context encased in single quotes. Here are some examples of good questions:
# """

# USER_PROMPT = """{CONTEXT}

# Question in single quotes:
# """
# def _send_llm_request(context : str) -> str:
    # "<FILL_IN>"

# COMMAND ----------

# ANSWER

# generate questions 
SYSTEM_PROMPT = """You are a journalist who is going to ask a Databricks usage question related to the provided context. Answer only with your question related to the provided context encased in single quotes. Here are some examples of good questions:
"""

USER_PROMPT = """{CONTEXT}

Question in single quotes:
"""
def _send_llm_request(context : str) -> str:
        messages = [
            ChatMessage(
                role=ChatMessageRole("system"),
                content=SYSTEM_PROMPT
            ),
            ChatMessage(
                role=ChatMessageRole("user"),
                content=USER_PROMPT.format(CONTEXT=context)
            )
        ]
        response = w.serving_endpoints.query(
            name=DBRX_ENDPOINT,
            messages=messages
        )
        return response.choices[0].message.content

# COMMAND ----------

# Now we will get some candidate chunks from our training dataset
candidates = (
    spark
    .table(f"{CATALOG}.{USER_SCHEMA}.gold_blog_chunks")
    .filter(F.col("chunk_num") == F.lit(0))
    .filter(F.col("char_length") < 1350)
)
candidates_pd = candidates.toPandas()
display(candidates_pd)

# COMMAND ----------

# # TODO:
# Using the candidate chunks and the llm prompt, use the .transform function to send a request to the Databricks endpoint and get the generated question back
# NOTE: This may take up to 5min to complete

# # Generate a question for each selected chunk
# candidates_pd["generated_question"] = "<FILL_IN>"

# # Remove the encased single quotes (')
# "<FILL_IN>"

# display(candidates_pd)

# COMMAND ----------

# ANSWER

# NOTE: This may take up to 5min to complete

# Generate a question for each selected chunk
candidates_pd["generated_question"] = candidates_pd["content"].transform(_send_llm_request)

# Remove the encased single quotes (')
candidates_pd["generated_question"] = candidates_pd["generated_question"].str.strip("'")
display(candidates_pd)

# COMMAND ----------

# Now we will split the dataframe into train and evaluation datasets
train_df = candidates_pd.sample(frac = 0.90, random_state = 42)
eval_df = candidates_pd.drop(train_df.index)

eval_df["uuid"] = eval_df["uuid"].transform(lambda x: [x])

def get_relevant_doc_ids(question : str) -> list[str]:
    docs = get_relevant_documents(question, index_name=VS_INDEX_FULL_NAME, k=10)
    return [_x["uuid"] for _x in docs]

eval_df["retrieved_docs"] = eval_df["generated_question"].transform(get_relevant_doc_ids)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Trying a New Embedding Model

# COMMAND ----------

# # TODO:

# We will use the `sentence-transformers` library to create embeddings for each text with the `all-MiniLM-L6-v2` model and then proceed to fine-tune the model
# model = "<FILL_IN>"

# COMMAND ----------

# ANSWER

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# COMMAND ----------

# Let's load the training and evaluation datasets using the right format and a training data data loader

from datasets import Dataset 

column_remap = {"content" : "anchor",  "generated_question" : "positive"}

train_df_copy = train_df
train_df_copy = train_df[["content", "generated_question"]].reset_index(drop=True)

eval_df_copy = eval_df
eval_df_copy = train_df[["content", "generated_question"]].reset_index(drop=True)

ft_train_dataset = Dataset.from_pandas(
    train_df_copy.rename(columns=column_remap)
)

ft_eval_dataset = Dataset.from_pandas(
    eval_df_copy.rename(columns=column_remap)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Fine-tuning the Embedding Model
# MAGIC
# MAGIC Note this can be done with an i3.xlarge

# COMMAND ----------

# # TODO:
# To improve the performance, let's fine-tune the model 

# # We'll use the MultipleNegativesRankingLoss from sentence-transformers

# train_loss = <FILL_IN>
# num_epochs = 1

# temp_checkpoint_dir = tempfile.TemporaryDirectory().name
# dbutils.fs.mkdirs(temp_checkpoint_dir)

# args = SentenceTransformerTrainingArguments(
#     <FILL_IN>
# )

# trainer = SentenceTransformerTrainer(
#     <FILL_IN>
# )

# COMMAND ----------

# ANSWER

train_loss = MultipleNegativesRankingLoss(model=model)
num_epochs = 1

temp_checkpoint_dir = tempfile.TemporaryDirectory().name
dbutils.fs.mkdirs(temp_checkpoint_dir)

args = SentenceTransformerTrainingArguments(
    # Required parameter:
    output_dir=temp_checkpoint_dir,
    # Optional training parameters:
    num_train_epochs=num_epochs,
    learning_rate=2e-5,
    auto_find_batch_size=True,
    warmup_ratio=0.1,
    # Optional tracking/debugging parameters:
    evaluation_strategy="steps",
    eval_steps=10,
    logging_steps=10
)

trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=ft_train_dataset,
    eval_dataset=ft_eval_dataset,
    loss=train_loss
)

# COMMAND ----------

with mlflow.start_run() as run:
    trainer.train()

# COMMAND ----------


