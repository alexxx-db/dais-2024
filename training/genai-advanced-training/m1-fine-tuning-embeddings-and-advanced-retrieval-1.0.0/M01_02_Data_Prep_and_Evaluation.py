# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning">
# MAGIC </div>
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC There are many powerful embedding models to choose from. Often, you'll find that an out-of-the-box (OOTB) embedding model performs perfectly well for your application, but in other cases you may find the performance is not up to snuff.
# MAGIC
# MAGIC  In the last notebook, you saw how we can embed unstructured text into a vector search index and learn how to retrieve documents based on queries. However, we hypothesize that a fine-tuned embedding model on PySpark documentation would yield better retrieval results. 
# MAGIC
# MAGIC  For fine-tuning, we need supervised dataset (question, answer) for the base model to learn from. In this notebook, you will learn how to prepare dataset for fine-tuning as well as generating benchmark evaluation metric.
# MAGIC
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) In this lesson you:<br>
# MAGIC  - Generate a dataset using an LLM to enable benchmarking
# MAGIC  - Use LLM-as-a-judge to gauge the quality of the generated dataset
# MAGIC  - Perform a benchmarking of your dataset
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %pip install --upgrade databricks-sdk==0.31.1 mlflow==2.14.1 tf-keras==2.16.0
# MAGIC dbutils.library.restartPython()
# MAGIC # databricks-sdk==0.24.0

# COMMAND ----------

# MAGIC %run ./Includes/Classroom-Setup

# COMMAND ----------

# MAGIC %md
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

VS_ENDPOINT = "adv_genai_course_endpoint_1" 
VS_INDEX = "spark_docs_gold_index"

DBRX_ENDPOINT = "databricks-dbrx-instruct"
LLAMA3_ENDPOINT = "databricks-meta-llama-3-1-70b-instruct"
GENERATED_QUESTIONS = "generated_questions"
GOLD_CHUNKS_FLAT = "spark_docs_gold_flat"

VS_INDEX_FULL_NAME = f"{SHARED_CATALOG}.{SHARED_SCHEMA}.{VS_INDEX}"

# COMMAND ----------

# MAGIC %md
# MAGIC # Read data

# COMMAND ----------

candidates = (
    spark
        .table(f"{CATALOG}.{USER_SCHEMA}.{GOLD_CHUNKS_FLAT}")
        .filter(F.col("chunk_num") == F.lit(0))
        .filter(F.col("char_length") > 450)
        .filter(~F.col("content").startswith("Source code"))
)

candidates_pd = candidates.toPandas()
display(candidates_pd)

# COMMAND ----------

# MAGIC %md
# MAGIC # Step 1: Prepare supervised data for fine-tuning
# MAGIC
# MAGIC Generate questions using LLM

# COMMAND ----------

SYSTEM_PROMPT = """You are a PySpark user who is going to ask a pyspark usage question related to the provided context. DO NOT use the function's name in your question. Answer only with your question related to the provided context encased in single quotes. Here are some examples of good questions:

Good Question in single quotes WITHOUT USING FUNCTION NAME:
'What function do I use to pick which columns to keep from my DataFrame?'

Good Question in single quotes WITHOUT USING FUNCTION NAME:
'How can I expand an array column to a single row per array entry?'
"""

USER_PROMPT = """{CONTEXT}

Good Question in single quotes WITHOUT USING FUNCTION NAME:
"""

# COMMAND ----------

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
        w = WorkspaceClient()
        response = w.serving_endpoints.query(
            name=DBRX_ENDPOINT,
            messages=messages
        )
        return response.choices[0].message.content

# COMMAND ----------

# MAGIC %md
# MAGIC The following cell might take ~6 mins to run.

# COMMAND ----------

# Generate a question for each selected chunk
candidates_pd["generated_question"] = candidates_pd["content"].transform(_send_llm_request)
# Remove the encased single quotes (')
candidates_pd["generated_question"] = candidates_pd["generated_question"].str.strip("'")

display(candidates_pd)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC You might be curious about the quality of the generated questions based on the documentation. Here, we can use LLM as a judge to determine the quality of the questions. To help the LLM assess the question quality, we can provide some example questions and scores to guide LLM to assign a score. 
# MAGIC
# MAGIC Below, we use a few example generated questions from the dataframe above.

# COMMAND ----------

example_context1 = candidates_pd[candidates_pd["content"].str.startswith("LinearDataGenerator")]["content"].values[0]

example_question1 = candidates_pd[candidates_pd["content"].str.startswith("LinearDataGenerator")]["generated_question"].values[0]

example1 = EvaluationExample(
    input=example_context1,
    output=(example_question1),
    score=3,
    justification=(
        "The question references 'provided utils'. The questions should be standalone. Additionally it uses a lot of exact terminology directly from the context."
    )
)

print(f"Example context 1: {example_context1}")
print("-"*120)
print(f"Example question 1: {example_question1}")

# COMMAND ----------

example_context2 = candidates_pd[candidates_pd["content"].str.startswith("pyspark.pandas.DataFrame.applymap")]["content"].values[0]
example_question2 = candidates_pd[candidates_pd["content"].str.startswith("pyspark.pandas.DataFrame.applymap")]["generated_question"].values[0]

example2 = EvaluationExample(
    input=example_context2,
    output=(example_question2),
    score=5,
    justification=(
        "The question is broad and doesn't directly reference the function name in the context."
    )
)

print(f"Example context 2: {example_context2}")
print("-"*120)
print(f"Example generated question 2: {example_question2}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Define custom metric
# MAGIC Now, we can define custom metrics to assess the quality of the generated questions. We will provide guidance to the LLM how to assign a score to each generated question. 
# MAGIC
# MAGIC In the grading prompt, we use a scale of 0-5, but this is completely up to the developer's discretion to decide an appropriate score range and rubric.

# COMMAND ----------

question_quality = make_genai_metric(
  name="GeneratedQuestionQuality",
  definition=(
      "Measures the quality of the LLM generated questions"),
  grading_prompt=(
      """Generated Question Quality: If a generated question seems to be too specific, containing lots of class names or method names we will give it a low score. If a generated question is broad and only contains a few class or method names we will rate it with a high score.
      - Score 1: This is not a question.
      - Score 2: This question is not relevant to PySpark. 
      - Score 3: This question is too specific and re-uses a lot of terminology from the context. It's not a common generic PySpark question. 
      - Score 4: The question is moderately specific and borrows minimal terminology from the context. 
      - Score 5: The question is open-ended and does not reference the exact funtion name in the context."""
  ),
  model=f"endpoints:/{LLAMA3_ENDPOINT}",
  examples=[example1, example2],
  parameters={"temperature": 0.0},
  aggregations=["mean", "variance"],
  greater_is_better=True,
)

# COMMAND ----------

eval_pd = candidates_pd.copy()
eval_pd["predictions"] = eval_pd["generated_question"]
eval_pd["inputs"] = eval_pd["content"]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Compute evaluation metrics using MLflow
# MAGIC The cell below could take ~10mins to complete.

# COMMAND ----------

with mlflow.start_run() as run:
    question_eval = mlflow.evaluate(
        data=eval_pd,
        model_type="question-answering",
        predictions="predictions",
        extra_metrics=[question_quality], # This now includes our custom metric!
      )

# COMMAND ----------

question_eval.metrics

# COMMAND ----------

# MAGIC %md
# MAGIC You can also view the metrics for each row in the dataframe.

# COMMAND ----------

eval_results_table = pd.DataFrame.from_dict(question_eval.tables["eval_results_table"])
eval_results_table

# COMMAND ----------

eval_results_table.groupby("GeneratedQuestionQuality/v1/score").count()["generated_question"]

# COMMAND ----------

# write out the generated questions and content table before we split into training and evaluation datasets
(
    spark
        .createDataFrame(candidates_pd)
        .select("uuid","generated_question","content")
        .write
        .mode("overwrite")
        .option("overwriteSchema","true")
        .saveAsTable(f"{CATALOG}.{USER_SCHEMA}.{GENERATED_QUESTIONS}")
)

# COMMAND ----------

# MAGIC %md
# MAGIC # Step 3: Generate baseline evaluation metrics
# MAGIC
# MAGIC Now that you have seen how we can use MLflow to generate evaluation metrics above, we can apply the same workflow below to establish baseline retrieval metric.

# COMMAND ----------

# split data into train and evaluation sets 
# we will use these datasets for fine-tuning our embedding model in the following notebook
generated_questions = (spark.table(f"{CATALOG}.{USER_SCHEMA}.{GENERATED_QUESTIONS}"))
train_df, eval_df = generated_questions.randomSplit([0.8,0.2], seed=41)
eval_pd = eval_df.toPandas()
eval_pd["uuid"] = eval_pd["uuid"].transform(lambda x: [x])
display(eval_pd)

# COMMAND ----------

def get_relevant_documents(question:str, index_name:str, k:int = 3, filters:str = None, max_retries:int = 3) -> List[dict]:
    """
    This function searches through the supplied vector index name and returns relevant documents 
    """
    # print(question)
    w = WorkspaceClient()
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

# write a function that we will apply across the entire evaluation dataset to get the relevant document IDs 
def get_relevant_doc_ids(question : str) -> list[str]:
    docs = get_relevant_documents(question, index_name=VS_INDEX_FULL_NAME, k=10)
    return [_x["uuid"] for _x in docs]

# test that it works for a single question
print(get_relevant_doc_ids("How can I read CSV files using Structured Streaming in PySpark?"))

# apply the function to the entire evaluation dataset
eval_pd["retrieved_docs"] = eval_pd["generated_question"].transform(get_relevant_doc_ids)

# COMMAND ----------

with mlflow.start_run() as run:
    eval_results = mlflow.evaluate(
        data = eval_pd,
        model_type="retriever",
        targets="uuid",
        predictions="retrieved_docs",
        evaluators="default",
        extra_metrics=[mlflow.metrics.recall_at_k(i) for i in range(1,10,1)]
    )

# COMMAND ----------

plt.plot([eval_results.metrics[f"recall_at_{i}/mean"] for i in range(1,10,1)], label="recall")
plt.title("Recall at k")
plt.xlabel("k")
plt.legend()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # Step 4: Write out train/eval sets
# MAGIC We write both train and eval dataframes out for fine-tuning in the next notebook

# COMMAND ----------

(
        train_df
            .write
            .mode("overwrite")
            .option("overwriteSchema","true")
            .saveAsTable(f"{CATALOG}.{USER_SCHEMA}.{GENERATED_QUESTIONS}_train")
)

(
        eval_df
            .write
            .mode("overwrite")
            .option("overwriteSchema","true")
        .saveAsTable(f"{CATALOG}.{USER_SCHEMA}.{GENERATED_QUESTIONS}_eval")
)

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
