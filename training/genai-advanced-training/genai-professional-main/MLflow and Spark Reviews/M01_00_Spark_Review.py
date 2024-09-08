# Databricks notebook source
# INCLUDE_HEADER_TRUE
# INCLUDE_FOOTER_TRUE

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC # Spark Review
# MAGIC This notebook serves as a quick overview of what Apache Spark is.
# MAGIC
# MAGIC
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) In this lesson you will:<br>
# MAGIC  - Create a Spark DataFrame
# MAGIC  - Analyze the Spark UI
# MAGIC  - Cache data
# MAGIC  - Convert between Pandas and PySpark DataFrames

# COMMAND ----------

# MAGIC %md
# MAGIC # What is Spark? 
# MAGIC
# MAGIC - Founded as a research project at UC Berkeley in 2009
# MAGIC - Open-source unified data analytics engine for big data
# MAGIC - Built-in APIs in SQL, Python, Scala, R, and Java
# MAGIC
# MAGIC ![](https://files.training.databricks.com/images/sparkcluster.png)

# COMMAND ----------

# MAGIC %md
# MAGIC # When to Use Spark?
# MAGIC 1. Scaling out 
# MAGIC    - Data or model is too large to process on a single machine, commonly resulting in out-of-memory errors
# MAGIC
# MAGIC 2. Speeding up 
# MAGIC    - Data or model is processing slowly and could benefit from shorter processing times and faster results
# MAGIC

# COMMAND ----------

# MAGIC %pip install Faker==0.7.4

# COMMAND ----------

# MAGIC %md 
# MAGIC ## PySpark DataFrame
# MAGIC
# MAGIC Let's start by creating a simple DataFrame in PySpark with two columns: id, v

# COMMAND ----------

from pyspark.sql.functions import col, rand

df = (spark.range(1, 1000000)
      .withColumn("id", (col("id") / 1000).cast("integer"))
      .withColumn("v", rand(seed=1)))

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC How does Spark help optimize workloads? 
# MAGIC
# MAGIC Let's take a step back and ask -- why were no Spark jobs kicked off above? Remember that Spark uses something called [lazy evaluation](https://spark.apache.org/docs/latest/rdd-programming-guide.html#rdd-operations). Since we didn't have to actually do anything to our data, Spark didn't need to execute anything across the cluster. We can kick-off some Spark Jobs by asking to display this DataFrame.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Understanding our DataFrame with Pyspark
# MAGIC
# MAGIC We can use some aggreagation functions to learn more about our DataFrame.

# COMMAND ----------

df.count()

# COMMAND ----------

# MAGIC %md 
# MAGIC # User-defined functions (UDFs)
# MAGIC
# MAGIC Now that you see we can use Python to leverage Apache Spark, you might wonder if you can write any functions to apply across a PySpark dataframe. The answer is a sounding yes! 
# MAGIC
# MAGIC User-defined functions (UDFs) are custom column transformation functions. 
# MAGIC - Row data is deserialized from Spark's native binary format to pass to the UDF, and the results are serialized back into Spark's native format
# MAGIC - There is additional interprocess communication overhead between the executor and a Python interpreter running on each worker node
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC Let's go through an example. We will first generate a list of fake emails and write a regular Python function that gets the first letter of the email. 

# COMMAND ----------

from faker import Faker

fake = Faker()
fake_email_list = [fake.company_email() for _ in range(100)]

# COMMAND ----------

# MAGIC %md ### Define a function
# MAGIC
# MAGIC Define a function (on the driver) to get the first letter of a string from the **`email`** field. 

# COMMAND ----------

def first_letter_function(email):
    return email[0]

first_letter_function(fake_email_list[0])

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Create and apply UDF
# MAGIC Register the function as a UDF. This serializes the function and sends it to executors to be able to transform DataFrame records.
# MAGIC
# MAGIC This example also uses Python type hints, which were introduced in Python 3.5. Type hints are not required for this example, but instead serve as "documentation" to help developers use the function correctly. They are used in this example to emphasize that the UDF processes one record at a time, taking a single str argument and returning a str value.
# MAGIC
# MAGIC

# COMMAND ----------

import pyspark.sql.functions as F

@F.udf("string")
def first_letter_udf(email: str) -> str:
    return email[0]

# COMMAND ----------

from pyspark.sql.types import StringType

fake_email_df = (spark.createDataFrame(fake_email_list, StringType())
                 .withColumnRenamed("value", "email")
                 .withColumn("first_letter", first_letter_udf("email"))
                 )
                 
display(fake_email_df)

# COMMAND ----------

# MAGIC %md
# MAGIC You just wrote your first UDF and applied it to a PySpark DataFrame! 
