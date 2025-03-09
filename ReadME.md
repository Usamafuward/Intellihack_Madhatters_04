# Stock Price Prediction Project

## Overview
This project develops a machine learning solution to predict stock prices 5 trading days into the future. The solution includes a complete data analysis pipeline, predictive modeling, and an end-to-end system design for production deployment.

## Project Structure
- `stock_prediction.py`: Main Python script containing the complete prediction workflow
- `future_predictions.csv`: CSV file with predictions for the test period
- `README.md`: This file
- `system_architecture_diagram.svg`: Visual diagram of the end-to-end system design

## Getting Started

### Prerequisites
- Python 3.8+
- Required Python packages:
  ```
  pandas
  numpy
  matplotlib
  seaborn
  scikit-learn
  xgboost
  lightgbm
  plotly
  ```

### Installation
1. Clone this repository
2. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```

### Running the Code
1. Place your stock data CSV in the project directory
2. Update the file path in the main function if needed
3. Run the script:
   ```
   python main.py
   ```

## Approach

### 1. Exploratory Data Analysis
- Analyzed historical price trends, trading volume, and returns
- Identified seasonality patterns and potential anomalies
- Examined correlation between different price metrics
- Visualized rolling statistics to understand volatility

### 2. Feature Engineering
- Created technical indicators:
  - Moving averages (SMA, EMA)
  - Relative Strength Index (RSI)
  - MACD (Moving Average Convergence Divergence)
  - Bollinger Bands
  - Momentum indicators
- Engineered calendar features
- Incorporated volume-based indicators
- Analyzed feature importance using correlation with target

### 3. Model Development
- Trained and evaluated multiple models:
  - Linear Regression
  - Random Forest
  - Gradient Boosting
  - XGBoost
  - LightGBM
- Used metrics including RMSE, MAE, R-squared, and directional accuracy
- Selected best model based on predictive performance
- Implemented a simulated trading strategy to validate practical value

### 4. Production System Design
- Designed an end-to-end system architecture for deploying the model
- Addressed data collection, processing, model operations, and insight delivery
- Considered scalability, reliability, latency, and cost requirements
- Identified potential implementation challenges and mitigation strategies

## Results

### Key Findings
- Technical indicators, especially moving averages and momentum features, showed strong predictive power
- The XGBoost model achieved the best performance, balancing accuracy and computational efficiency
- The trading strategy based on model predictions outperformed a simple buy-and-hold approach
- Time-series cross-validation provided more reliable performance estimates than standard random splitting

### Limitations
- The model's performance varies across different market conditions
- Limited availability of alternative data sources (news, sentiment, macroeconomic factors)
- Difficulty in predicting black swan events or unexpected market shocks
- Fixed prediction horizon (5 days) lacks flexibility for different trading strategies

### Future Improvements
- Incorporate alternative data sources for enhanced prediction
- Implement ensemble techniques combining multiple models
- Develop adaptive models that adjust to changing market conditions
- Create a reinforcement learning framework for optimized trading decisions
- Extend the prediction to multiple timeframes (1-day, 10-day, 30-day forecasts)

## References
- Atsalakis, G. S., & Valavanis, K. P. (2009). Surveying stock market forecasting techniques–Part II: Soft computing methods. Expert Systems with Applications, 36(3), 5932-5941.
- De Prado, M. L. (2018). Advances in financial machine learning. John Wiley & Sons.
- Sezer, O. B., Gudelek, M. U., & Ozbayoglu, A. M. (2020). Financial time series forecasting with deep learning: A systematic literature review: 2005–2019. Applied Soft Computing, 90, 106181.