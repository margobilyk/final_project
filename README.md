# Big Data Final Project — Chicago Crime Analysis and Predictive System

This repository contains the implementation of a complete big-data and machine-learning pipeline for analyzing crime in the City of Chicago.
The project includes an ETL pipeline based on the Medallion Architecture (Bronze → Silver → Gold), exploratory and statistical analysis, machine-learning model development, and deployment via FastAPI.

The work integrates large-scale data processing, spatiotemporal crime analysis, clustering, hypothesis testing, and operational ML serving.

## Repository Structure

```
final_project/
│
├── 01_bronze_load_validate/
│   └── Bronze_Load_Validate.ipynb
│
├── 02_silver_transform/
│   └── Silver_Clean_Transform.ipynb
│
├── 03_gold_analytics/
│   └── Gold_Analytics_Views.ipynb
│
├── ml_model/
│   ├── police_ops_auto.py
│   ├── main.py
│   └── police_model_auto.pkl
│
├── diagrams/
│   └── architecture.png
│
└── README.md
```

# 1. Project Overview

The objective of this project is to create a scalable, reproducible, and analysis-ready system capable of:

* Ingesting crime data from the Chicago Data Portal
* Processing the data through Bronze, Silver, and Gold layers using Spark and Delta Lake
* Performing spatiotemporal crime analysis
* Identifying structural crime patterns through clustering and hypothesis testing
* Training a predictive model for violent-crime risk estimation
* Deploying the model as an API to support operational decision-making

The full analytical and methodological details are described in the project documentation (BigData FP – Chicago Crimes). 

# 2. Medallion Architecture (Bronze → Silver → Gold)

### Bronze Layer: Raw Ingestion

* Batch ingestion of CSV data
* Standardization of column names
* Addition of metadata fields: `ingestion_timestamp`, `run_date`, `source_system`
* Storage in Delta format to preserve auditability and allow time-travel

### Silver Layer: Cleaning and Transformation

* Deduplication based on `ID` and `Case_Number`
* Geospatial filtering using a bounding box to remove invalid coordinates
* Standardization of categorical and textual fields
* Feature engineering including:

  * Time-of-day bins
  * Crime category grouping (Violent, Property, Other)
  * Cyclical transformations for hour and month
* Partitioning by District to improve query performance

### Gold Layer: Analytics and ML Preparation

* Aggregated crime statistics
* ML-ready datasets with engineered features
* Tables optimized for analysis and dashboarding


# 3. Exploratory and Statistical Analysis

The analysis includes several components derived from the processed Silver and Gold datasets, as described in the project report. 

### 3.1 Temporal Trends

Crime volumes exhibit seasonal patterns, with property-related incidents rising in the warmer months (May–August). Violent crime shows a steadier baseline throughout the year.

### 3.2 Spatiotemporal Heatmaps

The intersection of hour-of-day and day-of-week reveals a consistent high-intensity band between 12:00 and 23:00, especially on Fridays and Saturdays.
These patterns highlight time windows where additional staffing is most effective.

### 3.3 Arrest Efficiency vs. Crime Volume

There is little to no positive correlation between crime volume and arrest rate across districts. High-volume districts frequently show arrest rates below 15%, suggesting operational bottlenecks rather than purely resource shortages.


# 4. Advanced Analytics

### 4.1 District Clustering (K-Means)

A K-Means model (k=3) was used to segment districts by property crime count, violent crime count, and arrest rate.

Cluster types:

1. Low-risk districts with below-average crime
2. Districts with high ratios of violent to property crime
3. High-volume commercial districts with concentrated property crime

These clusters support differentiated resource-deployment strategies.

### 4.2 Hypothesis Testing

The widely assumed “weekend effect” was evaluated.
Daily crime totals on weekdays and weekends are statistically similar, but weekend crime is more temporally concentrated.
This indicates that staffing adjustments should target specific time windows rather than volume-based assumptions.


# 5. Machine Learning Model

A predictive model was built to estimate the probability of violent crime occurring in a specific district during a specific hour.

### Model

* Histogram-Based Gradient Boosting Classifier (`HistGradientBoostingClassifier`)
* Target: `Is_Violent` (binary)

### Feature Engineering (summarized from report table)

Features include:

* District (categorical encoding)
* Hour (sin/cos)
* Month (sin/cos)
* Day of week (integer)

These transformations capture temporal cycles and district-level variability. 

### Model Performance

* Recall for violent-crime detection: ~65%
* ROC-AUC: ~0.62
* F1 Score: 0.4581

The model is optimized for recall due to the safety-critical nature of the domain.


# 6. API Deployment (FastAPI)

The trained model is served using a FastAPI application.

### Starting the API

```
uvicorn main:app --reload
```

### Usage

Navigate to:

```
http://127.0.0.1:8000/docs
```

The interface allows users to submit district, hour, month, and day-of-week parameters and obtain a predicted violent-crime risk score.

# 7. Running the Entire Project

### Step 1: Run Databricks ETL Notebooks

In order:

1. Bronze_Load_Validate.ipynb
2. Silver_Clean_Transform.ipynb
3. Gold_Analytics_Views.ipynb

### Step 2: Train the Model Locally

```
python ml_model/police_ops_auto.py
```

### Step 3: Start the FastAPI Service

```
uvicorn ml_model.main:app --reload
```

# 8. Limitations

* Analysis relies on reported incidents only
* Some records excluded due to missing/invalid coordinates
* External factors such as weather or major events are not included
* Class imbalance impacts precision; the model prioritizes recall


# 9. Future Extensions

* Integration of streaming ingestion
* Incorporation of weather, event, and demographic data
* Additional clustering and anomaly-detection techniques
* Reinforcement-learning based patrol optimization
* Automated monitoring of model drift
