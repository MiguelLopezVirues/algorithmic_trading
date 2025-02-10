# 📈💻 Algorithmic Trading

<div style="text-align: center;">
  <img src="assets/algorithmic_trading.jpg" alt="Project Cover" />
</div>

## 📝 Project Overview

The **Algorithmic Trading** project aims to develop an automated trading system based on machine learning predictive models. By extracting, transforming, and modeling financial data from sources such as YahooFinance, FRED, and others, the goal is to eliminate emotions from trading activities and focus exclusively on quantitative risk management and trade optimization. 

The premise is simple: using probabilistic forecasting to establish prediction intervals that measure the uncertainty of the prediction, determining the probability that stock prices will grow or decrease. This allows for more confident establishment of educated Risk/Reward ratios for each trade. Based on these Risk/Reward ratios, more robust and statistically reproducible results can be expected.

Then to apply the strategies, a deployed ML model in AWS ECS once a week produces interval predictions every weekend during market close, forecasting prices for 5 steps ahead until Friday included and depositting them on a database in MongoDB. Upon scheduled market open times, adapted to each exchange, another service deployed un AWS Lambda reads predictions and taking the real-time price calculates a mean expected Risk/Reward ratio based on the upper and lower bound of the prediction. If a trade is open, those same upper and lower bound serve as dynamic Take Profit and Stop Loss for the trade, which in case of not reaching those limits, would close moments prior to market close on Friday.

---

### 🚀 Features

This project follows a structured approach through the following steps:

1. **Data Extraction:**
   - Collecting financial data from various sources such as YahooFinance, FRED, and other APIs.
   - Ensuring data quality and consistency for accurate modeling.

2. **Data Transformation:**
   - Cleaning and preprocessing the data to handle missing values, outliers, and inconsistencies.
   - Feature engineering to create meaningful predictors for the models.

3. **Model Development:**
   - Developing machine learning models for probabilistic forecasting.
   - Evaluating model configurations for different approaches: Direct multivariate, recursive multiseries.
   - Different feature combinations and feature selection
   - Hyperparameter tuning.
   - Assessment of probabilistic forcast interval coverage.

4. **Model Deployment:**
   - Deploying the model on AWS ECS to generate weekly predictions.
   - Storing predictions in a MongoDB database for real-time access.

5. **Trading Execution:**
   - Establishing trade Risk/Reward based on prediction bounds and open price. 
   - Using AWS Lambda to read predictions and execute trades based on calculated Risk/Reward ratios.
   - Implementing dynamic Take Profit and Stop Loss levels based on prediction intervals.


---

## 📁 Project Structure

```bash
Algorithmic-Trading/
├── assets/                     # Images or visual assets
├── data/                       # Raw data files and dictionaries
├── deployment/                 # Deployment-related files
├── notebooks/                  # Jupyter notebooks for analysis and modeling
│   ├── experiments
│   │   ├── 1_0_data_extraction.ipynb
│   │   ├── 1_2_first_analysis.ipynb
│   │   ├── 2_data_transformation.ipynb
│   │   ├── 3_data_load.ipynb
│   │   ├── 4_model_evaluation.ipynb
│   │   ├── 5_probability_forecast_and_simulation.ipynb
│   │   ├── 6_trading_API.ipynb
│   │   └── X_appendix-glossary_of_metrics.ipynb
├── src/                        # Python scripts for support functions
│   ├── __init__.py
│   │ 
│   ├── deployment
│   │   ├── __init__.py
│   │   ├── predict.py
│   │   └── train.py
│   │ 
│   ├── support
│   │   ├── __init__.py
│   │   ├── config
│   │   │   ├── countries_by_region.yaml
│   │   │   ├── indices_by_country.yaml
│   │   │   ├── tickers_to_include.yaml
│   │   │   └── countries_exchanges.py
│   │   ├── data_analysis.py
│   │   ├── data_extraction.py
│   │   ├── data_load.py
│   │   ├── data_transformation.py
│   │   ├── data_visualization.py
│   │   ├── file_handling.py
│   │   ├── model_evaluation.py
│   │   ├── timeseries_support.py
│   │   ├── trade_evaluation.py
│   │   └── utils.py
│   │
│   ├── trading
│   │   ├── __init__.py  
│   │   └── trading.py
│   │ 
│   └── data_etl_pipeline.py
│
├── Dockerfile
├── pyproject.toml
├── README.md
└── uv.lock
```

---

## 🛠️ Installation and Requirements

### Prerequisites

- Python >=3.10

#### **Core Packages**  
- **[polars](https://pola.rs/)** – High-performance DataFrame library for fast data processing.  
- **[pandas](https://pandas.pydata.org/docs/)** – Data manipulation and analysis.  
- **[numpy](https://numpy.org/doc/)** – Fundamental package for numerical computing.  

#### **Financial Data & Trading**  
- **[yfinance](https://github.com/ranaroussi/yfinance)** – Retrieves historical market data from Yahoo Finance.  
- **[TA-Lib](https://mrjbq7.github.io/ta-lib/)** – Technical analysis library for indicators and pattern recognition.  
- **[pandas-datareader](https://pandas-datareader.readthedocs.io/en/latest/)** – Extracts financial data from online sources, such as FRED.  
- **[pandas-market-calendars](https://pandas-market-calendars.readthedocs.io/en/latest/)** – Handles market trading calendars.  
- **[pandas-ta](https://github.com/twopirllc/pandas-ta)** – Technical analysis indicators for financial data.  


#### **Machine Learning & Forecasting**  
- **[scikit-learn](https://scikit-learn.org/stable/documentation.html)** – Machine learning algorithms for classification, regression, and clustering.  
- **[skforecast](https://github.com/JoaquinAmatRodrigo/skforecast)** – Framework for forecasting with scikit-learn regressors.  
- **[shap](https://shap.readthedocs.io/en/latest/)** – Model interpretability using SHAP values.  

#### **Database Storage**  
- **[boto3](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html)** – AWS SDK for Python.  
- **[pymongo](https://pymongo.readthedocs.io/en/stable/)** – MongoDB integration for Python.   

#### **Visualization & Interactive Tools**  
- **[matplotlib](https://matplotlib.org/stable/contents.html)** – Plotting and visualization.  
- **[seaborn](https://seaborn.pydata.org/)** – Statistical data visualization.  
- **[plotly](https://plotly.com/python/)** – Interactive visualizations.  


#### **Utilities & Performance Optimization**  
- **[joblib](https://joblib.readthedocs.io/en/latest/)** – Parallel computing and caching.  


### Installation Steps

1. Clone the repository:

   ```bash
   git clone https://github.com/username/algorithmic-trading
   cd algorithmic-trading
   ```

2. Install dependencies using Astral's UV:
   ```bash
   uv venv
   uv sync
   ```


---

## 🔄 Next Steps

- **Probabilistic interval callibration:**
  - Intervals obtained do not reflect the distribution of residuals in the test set.

- **Customization of Skforecast**
  - Implementing the use of exogenous features as dependent time series allows to use the most recent avaialble information. As opposed to being forced to use other time series as lagged exogenous features.
  - Extending available transformations and allowing to select which time series should be differenced.

- **Novel model architectures:**
  - Explore LSTMs and Temporal GNNs to predict quantiles or estimate deviations in a robust statistical manner.
  - Temporal GNNs with attention mechanisms can provide not only better performance, but enhanced explainability.

- **Complete Trading API Integration:**
  - Function to interact with the trading API and monitor records are available, a robust way to handle dynamic TP and SL is needed to complete integration.
  - Then deploy to AWS Lambda

- **Scalability:**
  - Explore scaling the system to handle more assets and higher frequencies.
  - Optimize the ETL pipeline for faster data processing.

---

## 🤝 Contributions

Contributions are welcome! Feel free to fork the repository, create pull requests, or raise issues for discussion.

---

## ✒️ Authors

- **Miguel López Virués** - [GitHub Profile](https://github.com/MiguelLópezVirués)

---

## 📜 License

This project is licensed under the MIT License.