# Databricks notebook source
# INCLUDE_HEADER_TRUE
# INCLUDE_FOOTER_TRUE

# COMMAND ----------

# MAGIC %md # Evaluating Embeddings for Retrieval
# MAGIC
# MAGIC There are many powerful embedding models to choose from. Often, you'll find that an out-of-the-box (OOTB) embedding model performs perfectly well for your application, but in other cases you may find the performance is not up to snuff.
# MAGIC
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) In this lesson you:<br>
# MAGIC  - Use a popular open source embedding model to embed your dataset into <a href="https://docs.databricks.com/en/generative-ai/vector-search.html" target="_blank">Mosaic AI Vector Search</a>
# MAGIC  - Generate a dataset using an LLM to enable benchmarking
# MAGIC  - Use LLM-as-a-judge to gauge the quality of the generated dataset
# MAGIC  - Perform a benchmarking of your dataset
# MAGIC
# MAGIC  In this module, we will see how we can improve document retrieval performance by fine-tuning embedding model. Our use case will leverage [PySpark documentation](https://spark.apache.org/docs/latest/api/python/index.html) as our dataset. Conventionally, we do not expect general embedding models to be able to work well on technical documentation, such as PySpark docs. Therefore, in this notebook, we will see how well we can retrieve relevant documentation based on users' submitted Spark questions. In the following notebook, we will fine-tune our out-of-the-box embedding model.
# MAGIC
# MAGIC  As a recap of the slides, the products used in this notebook include:
# MAGIC  - [Mosaic AI Vector Search](https://docs.databricks.com/en/generative-ai/vector-search.html)
# MAGIC  - [Mosaic AI Model Serving](https://docs.databricks.com/en/machine-learning/model-serving/index.html)
# MAGIC  - [Mosaic AI Foundation Model APIs](https://docs.databricks.com/en/machine-learning/foundation-models/index.html)

# COMMAND ----------

# MAGIC %md ## Install libraries

# COMMAND ----------

# MAGIC %pip install --upgrade databricks-sdk==0.24.0 mlflow==2.13.1
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./Includes/Classroom-Setup

# COMMAND ----------

# MAGIC %md ## Imports
# MAGIC
# MAGIC In the course, we will heavily leverage [Databricks SDK for Python](https://databricks-sdk-py.readthedocs.io/en/latest/clients/workspace.html).

# COMMAND ----------

import json
import pandas as pd
import time
from typing import List
import pyspark.sql.functions as F
from matplotlib import pyplot as plt

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole
from databricks.sdk.errors.platform import NotFound, ResourceDoesNotExist, BadRequest, PermissionDenied
from databricks.sdk.service.vectorsearch import EndpointType, VectorIndexType, DeltaSyncVectorIndexSpecResponse, EmbeddingSourceColumn, PipelineType

import mlflow
from mlflow.metrics.genai import make_genai_metric, EvaluationExample

# COMMAND ----------

# MAGIC %md ## Define user-specific constants

# COMMAND ----------

VS_ENDPOINT = "adv_genai_course_endpoint_1" 
VS_INDEX = "spark_docs_gold_index"

DBRX_ENDPOINT = "databricks-dbrx-instruct"
LLAMA3_ENDPOINT = "databricks-meta-llama-3-1-70b-instruct"
GENERATED_QUESTIONS = "generated_questions"
GOLD_CHUNKS_FLAT = "spark_docs_gold_flat"

VS_INDEX_FULL_NAME = f"{SHARED_CATALOG}.{SHARED_SCHEMA}.{VS_INDEX}"

# COMMAND ----------

# MAGIC %md
# MAGIC # Overview of a retrieval workflow 
# MAGIC
# MAGIC For any retrieval use cases, before we can get documents relevant to their queries, we need to complete the steps below in the following order: 
# MAGIC
# MAGIC 1. Identify text data to be ingested to vector search index
# MAGIC 1. Create a vector search endpoint 
# MAGIC 1. (Optional) Create an embedding model serving endpoint
# MAGIC     - This is only required if 
# MAGIC       - (i) your text data is not already converted into embeddings prior 
# MAGIC       - (ii) you are not using Databricks Foundation Model APIs since Databricks provides embedding model options out-of-the-box
# MAGIC     - In this notebook, we will skip this step because we will be using Databricks Foundation Model APIs
# MAGIC 1. Create a vector search index
# MAGIC 1. Finally, users can submit queries to retrieve relevant documents!

# COMMAND ----------

# MAGIC %md ## Step 1: Read data
# MAGIC Take a look at the dataset below. The data consists of PySpark documentation HTML pages which have been parsed and chunked. The columns are:
# MAGIC - filepath : the original path on dbfs to the html file
# MAGIC - content : the text or html (if it is a table) of the chunk
# MAGIC - category : what type of chunk it is
# MAGIC - char_length : how many characters the chunk is
# MAGIC - chunk_num : what number (starting from 0) chunk it is within the html file
# MAGIC - uuid : a unique identifier for each chunk 

# COMMAND ----------

display(spark.table(f"{CATALOG}.{USER_SCHEMA}.spark_docs_gold_flat"))

# COMMAND ----------

# MAGIC %md # Step 2: Create Vector Search endpoint using Mosaic AI Vector Search
# MAGIC
# MAGIC What is Mosaic AI Vector Search?
# MAGIC * Stores vector representation of your data, plus metadata
# MAGIC * Tightly integrated with Databricks Lakehouse
# MAGIC * Enables similarity search in real-time
# MAGIC * Allows filtering on metadata in search queries
# MAGIC * Can be queried using either REST API or Python client
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC
# MAGIC Below, we will first create the vector search endpoints. These should be created for you in the classroom environment, but if you run this in your own environment with the correct permissions this will still work.

# COMMAND ----------

w = WorkspaceClient()

def create_endpoint(endpoint:str, w:WorkspaceClient) -> None:
    """
    Creates an endpoint in the specified workspace.

    This function interacts with a given workspace client to create an endpoint at the specified location.
    
    Parameters:
    ----------
    endpoint : str
        The endpoint to be created.
        
    w : WorkspaceClient
        An instance of WorkspaceClient which provides the necessary methods and context to interact with the workspace where the endpoint will be created.
        
    Returns:
    -------
    None
        This function does not return any value. It performs the endpoint creation as a side effect.
    """
    try:
        w.vector_search_endpoints.get_endpoint(endpoint)
        print(f"Endpoint {endpoint} exists. Skipping endpoint creation.")
    except NotFound as e:
        print(f"Endpoint {endpoint} doesn't exist! Creating. May take up to 20 minutes. Note this block will fail if the endpoint isn't pre-created and you don't have the proper endpoint creation permissions.")
        w.vector_search_endpoints.create_endpoint(
            name=endpoint,
            endpoint_type=EndpointType("STANDARD")
        )

create_endpoint(VS_ENDPOINT, w)

# COMMAND ----------

# MAGIC %md Wait for the endpoints to be fully provisioned.

# COMMAND ----------

print(f"Checking the creation of {VS_ENDPOINT} and waiting for it to be provisioned.")
w.vector_search_endpoints.wait_get_endpoint_vector_search_endpoint_online(VS_ENDPOINT)
print("All endpoints created!")

# COMMAND ----------

# MAGIC %md # Step 3: Create vector search index 
# MAGIC
# MAGIC Check if the vector search index exists. If not, create it.
# MAGIC
# MAGIC Note that we do not need to have a separate step to create an embedding model serving endpoint because we are using Databricks Foundation Model APIs. As part of Databricks Foundation Model APIs, some models are readily available for users to query against. Refer to the [documentation](https://docs.databricks.com/en/machine-learning/foundation-models/index.html#pay-per-token-foundation-model-apis) to refer to the latest list of embedding models supported out-of-the-box. This is what we refer to as "Managed Embeddings". You can also host your own custom embedding model using [Provisioned Throughput Model Serving](https://docs.databricks.com/en/machine-learning/foundation-models/deploy-prov-throughput-foundation-model-apis.html) as well. 
# MAGIC
# MAGIC For more in-depth coverage of Mosaic AI Vector Search, check out our Generative AI Associate Course.

# COMMAND ----------

def create_index(endpoint:str, w:WorkspaceClient, sync_index:bool=False) -> None:
    """
    Creates an index in the specified endpoint.

    This function interacts with a given workspace client to create an index at the specified endpoint.
    
    Parameters:
    ----------
    endpoint : str
        The endpoint where the index should be created.
        
    w : WorkspaceClient
        An instance of WorkspaceClient which provides the necessary methods and context to interact with the workspace where the index will be created.

    sync_index : bool
        Whether the index should be synced if the endpoint already exists.
        
    Returns:
    -------
    None
        This function does not return any value. It performs the index creation as a side effect.
    """
    for index in w.vector_search_indexes.list_indexes(VS_ENDPOINT):
        if index.name == VS_INDEX_FULL_NAME.lower() and not sync_index:
            print(f"Found existing index in endpoint {endpoint}. Skipping index syncing.")
            return
    try:
        w.vector_search_indexes.sync_index(f"{SHARED_CATALOG}.{SHARED_SCHEMA}.{VS_INDEX}")
        print(f"Found existing index in endpoint {endpoint}. Synced index.")
    except ResourceDoesNotExist as e:
        print(f"Index in endpoint {endpoint} not found. Creating index.")
        try:
            w.vector_search_indexes.create_index(
                name=f"{SHARED_CATALOG}.{SHARED_SCHEMA}.{VS_INDEX}",
                endpoint_name=endpoint,
                primary_key="uuid",
                index_type=VectorIndexType("DELTA_SYNC"),
                delta_sync_index_spec=DeltaSyncVectorIndexSpecResponse(
                    embedding_source_columns=[
                        EmbeddingSourceColumn(
                            name="content",
                            embedding_model_endpoint_name="databricks-bge-large-en" # available via Foundation Model API
                        )],
                    pipeline_type=PipelineType("TRIGGERED"),
                    source_table=f"{CATALOG}.{USER_SCHEMA}.{GOLD_CHUNKS_FLAT}"
                )
            )
        except PermissionDenied as e:
            print(f"You do not have permission to create an index. Skipping this for this notebook. You'll create an index in the lab. {e}")
    except BadRequest as e:
            print(f"Index not ready to sync for endpoint {endpoint}. Skipping for now. {e}")


create_index(VS_ENDPOINT, w)

# COMMAND ----------

# MAGIC %md Wait until the endpoint is ready.
# MAGIC It may take ~6 mins for the vector index to be ready. You can also check the sync status under Compute -> Vector Search.

# COMMAND ----------

while w.vector_search_indexes.get_index(VS_INDEX_FULL_NAME).status.ready is not True:
    print("Vector search index is not ready yet...")
    time.sleep(30)

print("Vector search index is ready")

# COMMAND ----------

# MAGIC %md
# MAGIC # Step 4: User submits queries
# MAGIC
# MAGIC Define helper functions to retrieve relevant documents. 
# MAGIC
# MAGIC Notice that since we use Databricks Foundation Model APIs to manage embeddings, we do not need to convert text to embedding vectors for our queries. We can submit queries in plain text. The function below also shows that we can apply filters when searching against the vector search index.

# COMMAND ----------

def get_relevant_documents(question:str, index_name:str, k:int = 3, filters:str = None, max_retries:int = 3) -> List[dict]:
    """
    This function searches through the supplied vector index name and returns relevant documents 
    """
    docs = w.vector_search_indexes.query_index(
        index_name=index_name,
        columns=["uuid", "content", "category", "filepath"], # return these columns in the response
        filters_json=filters, # apply these filter statements in the query 
        num_results=k, # show k results in the response 
        query_text=question # query as text 
    )
    docs_pd = pd.DataFrame(docs.result.data_array)
    docs_pd.columns = [_c.name for _c in docs.manifest.columns]
    return json.loads(docs_pd.to_json(orient="records"))

# COMMAND ----------

answers = get_relevant_documents("How can I specify a list of columns to keep from my dataframe?", VS_INDEX_FULL_NAME, k=10)
answers

# COMMAND ----------

# the columns returned below are specified in the get_relevant_documents function above 
display(pd.DataFrame(answers))


# COMMAND ----------

# MAGIC %md
# MAGIC What if we would like to improve the retrieval performance? 
# MAGIC
# MAGIC We would need to prepare our data for the fine-tuning process. To do that, we need a supervised dataset where it consists of a question and also the reference documentation link. However, we do not have that data readily available currently. We also need a way to establish benchmark metric on our data while using the model as is. 
# MAGIC
# MAGIC Not to worry -- we can use LLM to generate user questions for us so that we can have a dataset of questions and answers (reference documentation links) for fine-tuning! Proceed to the next notebook to learn about data preparation and generating baseline (or benchmark) evaluation metrics. 
