# Stock Price Recommender

## Overview
The Stock Price Recommender is a comprehensive tool designed for analyzing and predicting stock prices. Utilizing various data analysis and machine learning techniques, this project aims to provide insights into stock market trends and assist in making informed investment decisions.

## Features
- **Data Collection**: Utilizes `yfinance` to gather historical stock data.
- **Exploratory Data Analysis (EDA)**: In-depth analysis of stock data, including trends in opening, closing, high, and low prices.
- **Stationarity Test**: Implements the ADF Fuller Test to check the stationarity of the data.
- **Modeling**: Includes both univariate and multivariate modeling notebooks to explore different aspects of stock price prediction.

## Installation
To set up the project environment, install the required packages using the following command:

```bash
pip install -r requirements.txt
```

## Usage
- **Data Analysis**: Explore the `analysis.ipynb` notebook for initial data analysis and visualization.
- **Univariate Modeling**: Check the `univariateModelling.ipynb` for univariate time series analysis.
- **Multivariate Modeling**: The `multivariateModelling.ipynb` notebook covers multivariate analysis.

## Datasets
- The project uses historical stock data, exemplified by `aapl.csv` during the time period of 2016-01-01 & 2024-01-01
