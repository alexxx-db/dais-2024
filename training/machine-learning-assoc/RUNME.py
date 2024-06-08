# Databricks notebook source
# MAGIC %md This notebook sets up the companion cluster(s) to run the solution accelerator. It also creates the Workflow to illustrate the order of execution. Happy exploring! 
# MAGIC 🎉
# MAGIC
# MAGIC **Steps**
# MAGIC 1. Simply attach this notebook to a cluster and hit Run-All for this notebook. A multi-step job and the clusters used in the job will be created for you and hyperlinks are printed on the last block of the notebook. 
# MAGIC
# MAGIC 2. Run the accelerator notebooks: Feel free to explore the multi-step job page and **run the Workflow**, or **run the notebooks interactively** with the cluster to see how this solution accelerator executes. 
# MAGIC
# MAGIC     2a. **Run the Workflow**: Navigate to the Workflow link and hit the `Run Now` 💥. 
# MAGIC   
# MAGIC     2b. **Run the notebooks interactively**: Attach the notebook with the cluster(s) created and execute as described in the `job_json['tasks']` below.
# MAGIC
# MAGIC **Prerequisites** 
# MAGIC 1. You need to have cluster creation permissions in this workspace.
# MAGIC
# MAGIC 2. In case the environment has cluster-policies that interfere with automated deployment, you may need to manually create the cluster in accordance with the workspace cluster policy. The `job_json` definition below still provides valuable information about the configuration these series of notebooks should run with. 
# MAGIC
# MAGIC **Notes**
# MAGIC 1. The pipelines, workflows and clusters created in this script are not user-specific. Keep in mind that rerunning this script again after modification resets them for other users too.
# MAGIC
# MAGIC 2. If the job execution fails, please confirm that you have set up other environment dependencies as specified in the accelerator notebooks. Accelerators may require the user to set up additional cloud infra or secrets to manage credentials. 

# COMMAND ----------

# MAGIC %pip install git+https://github.com/databricks-academy/dbacademy@v1.0.13 git+https://github.com/databricks-industry-solutions/notebook-solution-companion@safe-print-html --quiet --disable-pip-version-check

# COMMAND ----------

from solacc.companion import NotebookSolutionCompanion

# COMMAND ----------

job_json = {
  "name": "dais-2024-ml-assoc",
  "description": "DAIS FY25 ML Associate Courseware\n",
  "webhook_notifications": {},
  "timeout_seconds": 0,
  "max_concurrent_runs": 1,
  "tags": {
    "nannies": "STOP_DELETING_OUR_WORK",
    "policy": "DO_NOT_DELETE",
    "project": "DAIS FY25",
    "removeafter": "2024-12-31"
    },
    "run_as": {
    "user_name": "alex.barreto@databricks.com"
  },
  "tasks": [
    {
      "job_cluster_key": "ml-13-3-lts-cpu",
      "task_key": "launch-workflow",
      "run_if": "ALL_SUCCESS",
      "notebook_task": {
        "notebook_path": "00_data-preparation-for-machine-learning/Includes/_common",
        "source": "WORKSPACE"
      },
      "timeout_seconds": 0,
      "email_notifications": {},
      "webhook_notifications": {}
    },
    {
      "job_cluster_key": "ml-13-3-lts-cpu",
      "task_key": "lab-load-and-explore-data",
      "depends_on": [
        {
          "task_key": "launch-workflow"
        }
      ],
      "run_if": "ALL_SUCCESS",
      "notebook_task": {
        "notebook_path": "00_data-preparation-for-machine-learning/Solutions/01 - Managing and Exploring Data/1.LAB - Load and Explore Data",
        "source": "WORKSPACE"
      },
      "timeout_seconds": 0,
      "email_notifications": {},
      "webhook_notifications": {}
    },
    {
      "job_cluster_key": "ml-13-3-lts-cpu",
      "task_key": "lab-build-a-feature-engineering-pipeline",
      "depends_on": [
        {
          "task_key": "lab-load-and-explore-data"
        }
      ],
      "run_if": "ALL_SUCCESS",
      "notebook_task": {
        "notebook_path": "00_data-preparation-for-machine-learning/Solutions/02 - Data Preparation and Feature Engineering/2.LAB - Build a  Feature Engineering Pipeline",
        "source": "WORKSPACE"
      },
      "timeout_seconds": 0,
      "email_notifications": {},
      "webhook_notifications": {}
    },
    {
      "job_cluster_key": "ml-13-3-lts-cpu",
      "task_key": "lab-feature-engineering-with-feature-store",
      "depends_on": [
        {
          "task_key": "lab-build-a-feature-engineering-pipeline"
        }
      ],
      "run_if": "ALL_SUCCESS",
      "notebook_task": {
        "notebook_path": "00_data-preparation-for-machine-learning/Solutions/03 - Feature Store/3.LAB - Feature Engineering with Feature Store",
        "source": "WORKSPACE"
      },
      "timeout_seconds": 0,
      "email_notifications": {},
      "webhook_notifications": {}
    },
    {
      "job_cluster_key": "ml-13-3-lts-cpu",
      "task_key": "lab-model-development-tracking-mlflow",
      "depends_on": [
        {
          "task_key": "lab-feature-engineering-with-feature-store"
        }
      ],
      "run_if": "ALL_SUCCESS",
      "notebook_task": {
        "notebook_path": "01_machine-learning-model-development/01 - Model Development Workflow/1.LAB - Model Development Tracking with MLflow",
        "source": "WORKSPACE"
      },
      "timeout_seconds": 0,
      "email_notifications": {},
      "webhook_notifications": {}
    },
    {
      "job_cluster_key": "ml-13-3-lts-cpu",
      "task_key": "lab-hyperparameter-tuning-with-hyperopt",
      "depends_on": [
        {
          "task_key": "lab-model-development-tracking-mlflow"
        }
      ],
      "run_if": "ALL_SUCCESS",
      "notebook_task": {
        "notebook_path": "01_machine-learning-model-development/Solutions/02 - Hyperparameter Tuning/2.LAB - Hyperparameter Tuning with Hyperopt",
        "source": "WORKSPACE"
      },
      "timeout_seconds": 0,
      "email_notifications": {},
      "webhook_notifications": {}
    },
    {
      "job_cluster_key": "ml-13-3-lts-cpu",
      "task_key": "lab-automl",
      "depends_on": [
        {
          "task_key": "lab-hyperparameter-tuning-with-hyperopt"
        }
      ],
      "run_if": "ALL_SUCCESS",
      "notebook_task": {
        "notebook_path": "01_machine-learning-model-development/Solutions/03 - AutoML/3.LAB - AutoML",
        "source": "WORKSPACE"
      },
      "timeout_seconds": 0,
      "email_notifications": {},
      "webhook_notifications": {}
    },
    {
      "job_cluster_key": "ml-13-3-lts-cpu",
      "task_key": "lab-batch-deployment",
      "depends_on": [
        {
          "task_key": "lab-automl"
        }
      ],
      "run_if": "ALL_SUCCESS",
      "notebook_task": {
        "notebook_path": "02_machine-learning-model-deployment/Solutions/02 - Batch Deployment/2.LAB - Batch Deployment",
        "source": "WORKSPACE"
      },
      "timeout_seconds": 0,
      "email_notifications": {},
      "webhook_notifications": {}
    },
    {
      "job_cluster_key": "ml-13-3-lts-cpu",
      "task_key": "lab-realtime-deployment",
      "depends_on": [
        {
          "task_key": "lab-batch-deployment"
        }
      ],
      "run_if": "ALL_SUCCESS",
      "notebook_task": {
        "notebook_path": "02_machine-learning-model-deployment/Solutions/04 - Real-time Deployment/4.LAB - Real-time Deployment with Model Serving",
        "source": "WORKSPACE"
      },      
      "timeout_seconds": 0,
      "email_notifications": {},
      "webhook_notifications": {}
    },
    {
      "job_cluster_key": "ml-13-3-lts-cpu",
      "task_key": "load-and-explore-data",
      "depends_on": [
        {
          "task_key": "launch-workflow"
        }
      ],
      "run_if": "ALL_SUCCESS",
      "notebook_task": {
        "notebook_path": "00_data-preparation-for-machine-learning/Solutions/01 - Managing and Exploring Data/1.1 - Load and Explore Data",
        "source": "WORKSPACE"
      },
      "timeout_seconds": 0,
      "email_notifications": {},
      "webhook_notifications": {}
    },
    {
      "job_cluster_key": "ml-13-3-lts-cpu",
      "task_key": "data-imputation-and-transformation-pipeline",
      "depends_on": [
        {
          "task_key": "load-and-explore-data"
        }
      ],
      "run_if": "ALL_SUCCESS",
      "notebook_task": {
        "notebook_path": "00_data-preparation-for-machine-learning/Solutions/02 - Data Preparation and Feature Engineering/2.1 - Data Imputation and Transformation Pipeline",
        "source": "WORKSPACE"
      },
      "timeout_seconds": 0,
      "email_notifications": {},
      "webhook_notifications": {}
    },
    {
      "job_cluster_key": "ml-13-3-lts-cpu",
      "task_key": "build-a-feature-engineering-pipeline",
      "depends_on": [
        {
          "task_key": "data-imputation-and-transformation-pipeline"
        }
      ],
      "run_if": "ALL_SUCCESS",
      "notebook_task": {
        "notebook_path": "00_data-preparation-for-machine-learning/Solutions/02 - Data Preparation and Feature Engineering/2.2 - Build a Feature Engineering Pipeline",
        "source": "WORKSPACE"
      },
      "timeout_seconds": 0,
      "email_notifications": {},
      "webhook_notifications": {}
    },
    {
      "job_cluster_key": "ml-13-3-lts-cpu",
      "task_key": "using-a-feature-store-for-feature-engineering",
      "depends_on": [
        {
          "task_key": "build-a-feature-engineering-pipeline"
        }
      ],
      "run_if": "ALL_SUCCESS",
      "notebook_task": {
        "notebook_path": "00_data-preparation-for-machine-learning/Solutions/03 - Feature Store/3.1 - Using Feature Store for Feature Engineering",
        "source": "WORKSPACE"
      },
      "timeout_seconds": 0,
      "email_notifications": {},
      "webhook_notifications": {}
    },
    {
      "job_cluster_key": "ml-13-3-lts-cpu",
      "task_key": "training-regression-models",
      "depends_on": [
        {
          "task_key": "using-a-feature-store-for-feature-engineering"
        }
      ],
      "run_if": "ALL_SUCCESS",
      "notebook_task": {
        "notebook_path": "01_machine-learning-model-development/Solutions/01 - Model Development Workflow/1.1a - Training Regression Models",
        "source": "WORKSPACE"
      },  
      "timeout_seconds": 0,
      "email_notifications": {},
      "webhook_notifications": {}
    },
    {
      "job_cluster_key": "ml-13-3-lts-cpu",
      "task_key": "training-classification-models",
      "depends_on": [
        {
          "task_key": "training-regression-models"
        }
      ],
      "run_if": "ALL_SUCCESS",
      "notebook_task": {
        "notebook_path": "01_machine-learning-model-development/Solutions/01 - Model Development Workflow/1.1b - Training Classification Models",
        "source": "WORKSPACE"
      },
      "timeout_seconds": 0,
      "email_notifications": {},
      "webhook_notifications": {}
    },
    {
      "job_cluster_key": "ml-13-3-lts-cpu",
      "task_key": "model-tracking-with-mlflow",
      "depends_on": [
        {
          "task_key": "training-classification-models"
        }
      ],
      "run_if": "ALL_SUCCESS",
      "notebook_task": {
        "notebook_path": "01_machine-learning-model-development/Solutions/01 - Model Development Workflow/1.2 - Model Tracking with MLflow",
        "source": "WORKSPACE"
      },
      "timeout_seconds": 0,
      "email_notifications": {},
      "webhook_notifications": {}
    },
    {
      "job_cluster_key": "ml-13-3-lts-cpu",
      "task_key": "hyperparameter-tuning-with-hyperopt",
      "depends_on": [
        {
          "task_key": "model-tracking-with-mlflow"
        }
      ],
      "run_if": "ALL_SUCCESS",
      "notebook_task": {
        "notebook_path": "01_machine-learning-model-development/Solutions/02 - Hyperparameter Tuning/2.1 - Hyperparameter Tuning with Hyperopt",
        "source": "WORKSPACE"
      },      
      "timeout_seconds": 0,
      "email_notifications": {},
      "webhook_notifications": {}
    },
    {
      "job_cluster_key": "ml-13-3-lts-cpu",
      "task_key": "automated-model-dvpt-with-automl",
      "depends_on": [
        {
          "task_key": "hyperparameter-tuning-with-hyperopt"
        }
      ],
      "run_if": "ALL_SUCCESS",
      "notebook_task": {
        "notebook_path": "01_machine-learning-model-development/Solutions/03 - AutoML/3.1 - Automated Model Development with AutoML",
        "source": "WORKSPACE"
      },
      "timeout_seconds": 0,
      "email_notifications": {},
      "webhook_notifications": {}
    },
    {
      "job_cluster_key": "ml-13-3-lts-cpu",
      "task_key": "batch-deployment",
      "depends_on": [
        {
          "task_key": "automated-model-dvpt-with-automl"
        }
      ],
      "run_if": "ALL_SUCCESS",
      "notebook_task": {
        "notebook_path": "02_machine-learning-model-deployment/Solutions/02 - Batch Deployment/2.1 - Batch Deployment",
        "source": "WORKSPACE"
      },
      "timeout_seconds": 0,
      "email_notifications": {},
      "webhook_notifications": {}
    },
    {
      "job_cluster_key": "ml-13-3-lts-cpu",
      "task_key": "pipeline-deployment",
      "depends_on": [
        {
          "task_key": "batch-deployment"
        }
      ],
      "run_if": "ALL_SUCCESS",
      "notebook_task": {
        "notebook_path": "02_machine-learning-model-deployment/Solutions/03 - Pipeline Deployment/3.1.a - Pipeline Deployment",
        "source": "WORKSPACE"
      },
      "timeout_seconds": 0,
      "email_notifications": {},
      "webhook_notifications": {}
    },
    {
      "job_cluster_key": "ml-13-3-lts-cpu",
      "task_key": "realtime-deployment-with-model-serving",
      "depends_on": [
        {
          "task_key": "batch-deployment"
        }
      ],
      "run_if": "ALL_SUCCESS",
      "notebook_task": {
        "notebook_path": "02_machine-learning-model-deployment/Solutions/04 - Real-time Deployment/4.1 - Real-time Deployment with Model Serving",
        "source": "WORKSPACE"
      },
      "timeout_seconds": 0,
      "email_notifications": {},
      "webhook_notifications": {}
    },
    {
      "job_cluster_key": "ml-13-3-lts-cpu",
      "task_key": "custom-model-deployment-with-model-serving",
      "depends_on": [
        {
          "task_key": "realtime-deployment-with-model-serving"
        }
      ],
      "run_if": "ALL_SUCCESS",
      "notebook_task": {
        "notebook_path": "02_machine-learning-model-deployment/Solutions/04 - Real-time Deployment/4.2 - Custom Model Deployment with Model Serving",
        "source": "WORKSPACE"
      },
      "timeout_seconds": 0,
      "email_notifications": {},
      "webhook_notifications": {}
    }
  ],
  "job_clusters": [
    {
      "job_cluster_key": "ml-13-3-lts-cpu",
      "new_cluster": {
        "spark_version": "13.3.x-cpu-ml-scala2.12",
        "num_workers": 1,
        "node_type_id": {"AWS": "i4i.4xlarge", "MSA": "Standard_L16as_v3", "GCP": "a2-highgpu-1g"}, 
        "custom_tags": {
          "usage": "solacc_testing"
        },
      }
    }
  ]
}

# COMMAND ----------

dbutils.widgets.dropdown("run_job", "False", ["True", "False"])
run_job = dbutils.widgets.get("run_job") == "True"
nsc = NotebookSolutionCompanion()
nsc.deploy_compute(job_json, run_job=run_job)

# COMMAND ----------


