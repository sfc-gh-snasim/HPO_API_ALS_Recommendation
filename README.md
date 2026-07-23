# HPO_API_ALS_Recommendation

## Overview

This project builds a collaborative filtering recommendation engine using the **Alternating Least Squares (ALS)** algorithm from the `implicit` library. It leverages Snowflake's **Hyperparameter Optimization (HPO) API** with multi-node Container Runtime to scale model tuning across a grid of 81 hyperparameter combinations — including rank, regularization, iterations, and confidence scaling (alpha). Performance is evaluated using Precision@10, MAP@10, and NDCG@10.

The notebook demonstrates single-node vs. multi-node HPO speedup comparison, Snowflake Experiment Tracking for all trials, and registration of the best model as a CustomModel in the Snowflake Model Registry for SQL-callable inference.

## Architecture

```
MovieLens 1M Dataset
       │
       ▼
┌─────────────────────┐
│  Data Preparation   │  80/20 temporal split, sparse matrix indexing
└─────────────────────┘
       │
       ▼
┌─────────────────────┐
│  HPO (Grid Search)  │  81 trials (3×3×3×3 grid)
│  Phase 1: 1 node    │  max_concurrent_trials=2
│  Phase 2: 4 nodes   │  max_concurrent_trials=11
└─────────────────────┘
       │
       ▼
┌─────────────────────┐
│  Experiment Tracking│  All 81 trials + best model logged
└─────────────────────┘
       │
       ▼
┌─────────────────────┐
│  Model Registry     │  CustomModel with embedded ID mappings
│  (SPCS target)      │  SQL-callable: model!recommend()
└─────────────────────┘
```

## Hyperparameter Search Space

| Parameter | Values | Description |
|-----------|--------|-------------|
| RANK | 20, 50, 100 | Number of latent factors |
| REGPARAM | 0.01, 0.05, 0.1 | L2 regularization |
| MAXITER | 10, 20, 30 | Training iterations |
| ALPHA | 1.0, 10.0, 40.0 | Confidence scaling factor |

## Prerequisites

- Snowflake account with Container Runtime access
- Compute pool with multi-node capability (MIN_NODES >= 4 recommended)
- External access integration for pip installs and dataset download

## Setup

### 1. Create a Git Repo Workspace

1. In Snowsight, navigate to **Workspaces**.
2. Click **+ New** → **Git Repository**.
3. Enter the repository URL:
   ```
   https://github.com/sfc-gh-snasim/HPO_API_ALS_Recommendation.git
   ```
4. Configure authentication if the repo is private, otherwise proceed with public access.
5. Click **Create** to clone the repository into your workspace.

### 2. Copy Code to Your Workspace

1. Open the newly created Git repo workspace.
2. All project files — including the notebook `ALS_RECOMMENDATION_MODEL.ipynb` — will be available in the workspace file tree.
3. If you prefer to work in a separate workspace, create a new workspace and copy the files from the git repo workspace into it.

### 3. Run the Notebook

1. Open `ALS_RECOMMENDATION_MODEL.ipynb`.
2. Create or connect to a compute service (multi-node compute pool recommended).
3. Run all cells sequentially to:
   - Install dependencies (`implicit`, `scipy`)
   - Download and load the MovieLens 1M dataset (~1M ratings, 6K users, 3.7K items)
   - Prepare train/test split (80/20 temporal) with precomputed sparse matrix indices
   - Define the ALS training function with ranking metric evaluation (P@10, MAP@10, NDCG@10)
   - Run single-node HPO (81 trials, 2 concurrent) as baseline
   - Scale cluster to 4 nodes and run multi-node HPO (81 trials, 11 concurrent)
   - Compare speedup between single-node and multi-node runs
   - Log all trials and best model metrics to Snowflake Experiment Tracking
   - Register best model as a CustomModel in the Snowflake Model Registry
   - Run sample inference with real user IDs

## Notebook Sections

1. **Setup and Imports** — Install packages, import Snowflake ML + implicit libraries, create database/schema
2. **Data Loading** — Download MovieLens 1M, load into Snowflake table, prepare DataConnectors with index mappings
3. **Training Function** — ALS model training with `implicit.als.AlternatingLeastSquares`, evaluation via `implicit.evaluation.ranking_metrics_at_k`
4. **Single-Node HPO** — Baseline run on 1 node with 2 concurrent trials
5. **Multi-Node HPO** — Scale to 4 nodes, run with 11 concurrent trials, compare speedup
6. **Results & Experiment Tracking** — Extract best model/config, log all trials to Snowflake Experiments
7. **Model Registry & Inference** — Wrap best model as CustomModel (with embedded user/item ID mappings), register with SPCS target, run sample recommendations

## Key Snowflake APIs Used

- `snowflake.ml.modeling.tune.Tuner` / `TunerConfig` / `GridSearch` — HPO framework
- `snowflake.ml.runtime_cluster.scale_cluster` / `get_nodes` — Multi-node cluster management
- `snowflake.ml.data.DataConnector` — Data loading for distributed training
- `snowflake.ml.experiment.ExperimentTracking` — Experiment logging
- `snowflake.ml.registry.Registry` — Model registration
- `snowflake.ml.model.custom_model.CustomModel` — Custom inference wrapper