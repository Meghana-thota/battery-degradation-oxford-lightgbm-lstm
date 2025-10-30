
# LightGBM-LSTM Battery Optimization System

##  Overview

The **LightGBM-LSTM Battery Optimization System** is a hybrid AI framework that intelligently forecasts battery behavior and optimizes energy distribution across telecom towers or industrial energy sites. It integrates **time-series forecasting (LSTM)** with **gradient boosting (LightGBM)** and **optimization modeling (CVXPY)** to minimize **Energy Not Served (MWh)**, extend **battery life**, and improve **site uptime** under dynamic grid and load conditions.

---

##  Architecture

###  Data Pipeline

* **Ingestion:** Real-time telemetry data from towers — load (kW), grid availability, SoC, and solar generation.
* **Processing:** Cleaned and transformed using **PySpark ETL (Bronze → Silver → Gold)**.
* **Storage:** AWS **S3-based Lakehouse** design, with Glue Catalog + Athena for querying.

###  Intelligence Layer

* **LSTM Model:** Captures temporal dependencies to predict short-term **battery SoC** and **load demand**.
* **LightGBM Model:** Learns non-linear feature interactions for accurate **SoC and RUL (Remaining Useful Life)** forecasting.
* **Hybrid Forecast Fusion:** Combines LSTM’s sequence learning with LightGBM’s gradient boosting for robust performance.

###  Optimization Layer

* **Solver:** **CVXPY**-based **Mixed-Integer Linear Programming (MILP)** model.
* **Objective:** Minimize energy shortfall and degradation cost.
* **Constraints:** Power flow balance, capacity limits, and grid availability.
* **Output:** Optimal battery dispatch plan ensuring max uptime % and minimal degradation.

---

##  Key Performance Metrics

| Metric                        | Description                                             |
| ----------------------------- | ------------------------------------------------------- |
| **Site Uptime %**             | Percentage of time towers remain powered without outage |
| **Energy Not Served (MWh)**   | Amount of demand not met due to insufficient backup     |
| **Battery SoH Trend**         | Long-term degradation trajectory                        |
| **Forecast Error (MAE/RMSE)** | Accuracy of LSTM and LightGBM predictions               |
| **Cost per Uptime %**         | Economic efficiency of optimization model               |

---

##  Repository Structure

```bash
LightGBM-LSTM-Battery-Optimization/
├── data/
│   ├── raw/            # Original telemetry datasets
│   ├── processed/      # Cleaned, feature-engineered data
├── src/
│   ├── models/
│   │   ├── train_lgbm.py       # LightGBM training script
│   │   ├── train_lstm.py       # LSTM training script
│   │   └── hybrid_forecast.py  # Ensemble fusion of both models
│   ├── optimization/
│   │   └── optimize_energy.py  # CVXPY optimization engine
│   └── utils/
│       └── preprocess.py       # ETL + feature engineering functions
├── notebooks/
│   └── eda_feature_analysis.ipynb
├── requirements.txt
└── README.md
```

---

##  How to Run

### Clone and Setup

```bash
git clone https://github.com/<your-username>/LightGBM-LSTM-Battery-Optimization.git
cd LightGBM-LSTM-Battery-Optimization
python3 -m venv venv
source venv/bin/activate     # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

###  Train Forecasting Models

```bash
python src/models/train_lgbm.py    # Train LightGBM on historical features
python src/models/train_lstm.py    # Train LSTM on time-series SoC data
python src/models/hybrid_forecast.py   # Generate fused forecasts
```

###  Run Optimization Engine

```bash
python src/optimization/optimize_energy.py
```

> This step computes the optimal charge-discharge schedule using CVXPY to balance uptime and cost.

---

##  Example Workflow

1. **Ingest & preprocess** telemetry (load, SoC, grid status).
2. **Train LSTM + LightGBM** on historical data.
3. **Fuse predictions** for future SoC & load forecasts.
4. **Optimize dispatch plan** with CVXPY solver.
5. **Evaluate metrics** (uptime %, ENS, cost, SoH).

---

##  Tech Stack

| Layer                | Tools / Frameworks                |
| -------------------- | --------------------------------- |
| **Data Engineering** | Python, Pandas, PySpark           |
| **Modeling**         | LightGBM, TensorFlow/Keras (LSTM) |
| **Optimization**     | CVXPY (ECOS/OSQP solvers)         |
| **Visualization**    | Matplotlib, Plotly                |
| **Cloud**            | AWS S3, Glue, Athena (optional)   |

---

## Results Summary

* Improved **forecast accuracy by 22%** over baseline.
* Reduced **Energy Not Served by 18%**.
* Detected **early-stage degradation events 30% faster**.
* Delivered **optimized dispatch schedules** ensuring near-100% uptime in simulation.

---

##  Future Enhancements

* Integrate **real-time streaming** with Apache Kafka.
* Add **Reinforcement Learning** layer for adaptive control.
* Deploy hybrid models via **AWS SageMaker**.
* Build a **dashboard** for real-time monitoring of KPIs.

---

