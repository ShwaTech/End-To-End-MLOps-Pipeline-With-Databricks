# End-To-End-MLOps-Pipeline-With-Databricks

An End-To-End MLOps pipeline using the Marvel Characters dataset. It covers everything from raw data ingestion, through model training and feature engineering, to deployment, monitoring, and CI/CD. The pipeline leverages a suite of modern tools and best practices to ensure reproducibility, maintainability, and scalability.

## Project Overview

This project is designed to illustrate how to build and manage a full machine learning pipeline with real-world components. It includes:

Data ingestion, cleaning, and feature engineering

Model training, evaluation, and versioning

Deployment of the model to a serving endpoint (e.g., Databricks)

Monitoring of deployed models, dashboards, drift detection, and scheduled refreshes

Automation via CI/CD workflows, environment & dependency management, reproducible builds

The goal is to provide a template that developers, ML engineers, and data scientists can adapt for their own projects.

## Project Structure

```bash
├── .github/
│    └── workflows/                               # CI/CD workflows (GitHub Actions) 
│        ├── ci.yaml                              # CI YAML Files
│        └── cd.yaml                              # CD YAML Files
├── data/                                         # Raw or processed datasets  
│   └── marvel_characters_dataset.csv
├── notebooks/                                    # Exploratory data analysis & prototyping notebooks  
│   └── Databricks notebooks for exploration / prototyping
├── resources/                                    # Auxiliary resources  
│   ├── bundle_monitoring.yml
│   └── model_deployment.yml
├── scripts/                                      # Stand-alone scripts for discrete pipeline tasks  
│   └── stand-alone pipeline task scripts
├── src/
│   └── marvel_characters/                        # Core package module
│       ├── __init__.py
│       ├── data_processing.py                    # Data loading, cleaning, feature engineering
│       ├── model_training.py                     # Training logic, model definitions
│       ├── utils.py                              # Utility functions
│       └── …                                     # Other modules (e.g. evaluation, inference)
├── tests/                                        # Unit / integration tests  
│   ├── marvel_characters/                        # Core package module
│   │   ├── __init__.py
│   │   └── test_data_processor.py                # Testing Data Preprocessor.
│   ├── __init__.py
│   └── conftest.py
├── .pre-commit-config.yaml                       # Pre-commit hooks (linting, formatting, etc.)  
├── databricks.yml                                # Configuration for Databricks deployment / clusters  
├── project_config_marvel.yml                     # Project-specific settings  
├── pyproject.toml                                # Project metadata & dependencies  
├── uv.lock                                       # (auto-generated) locks dependency versions  
├── version.txt                                   # Project version  
└── README.md                                     # Project documentation (you’re reading it)  
```

## Clone The Repository

```bash
git clone git@github.com:ShwaTech/End-To-End-MLOps-Pipeline-With-Databricks.git

cd End-To-End-MLOps-Pipeline-With-Databricks
```

## Ensure you have **uv** installed

```bash
pip install uv
```

## Initialize and sync the project environment

```bash
uv sync --extra dev
```

This will: Create a virtual environment (.venv/)

## Activate environment

```bash
source .venv/bin/activate  # Linux/Mac

.venv\Scripts\activate     # Windows
```

## Data

Using the [**Marvel Characters Dataset**](https://www.kaggle.com/datasets/mohitbansal31s/marvel-characters?resource=download) from Kaggle.

This dataset contains detailed information about Marvel characters (e.g., name, powers, physical attributes, alignment, etc.).
It is used to build classification and feature engineering models for various MLOps tasks, such as predicting character attributes or status.

## Scripts

- `01_process_data.py`: Loads and preprocesses the Marvel dataset, splits into train/test, and saves to the catalog.
- `02_train_register_custom_model.py`: Performs feature engineering and trains the Marvel character model.
- `03_deploy_model.py`: Deploys the trained Marvel model to a Databricks model serving endpoint.
- `04_post_commit_status.py`: Posts status updates for Marvel integration tests to GitHub.
- `05_refresh_monitor.py`: Refreshes monitoring tables and dashboards for Marvel model serving.
