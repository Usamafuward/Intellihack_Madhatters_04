# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# For data preprocessing and feature engineering
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Machine learning models
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
import xgboost as xgb
from lightgbm import LGBMRegressor

# 1. Data Loading and Initial Exploration
def load_and_explore_data(file_path):
    """
    Load stock data from CSV and perform initial exploration
    """
    # Load the data
    df = pd.read_csv(file_path)
    
    # Convert date column to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Set date as index for time series analysis
    df.set_index('Date', inplace=True)
    
    # Sort by date to ensure chronological order
    df = df.sort_index()
    
    # Display basic information
    print(f"Dataset Shape: {df.shape}")
    print("\nFirst 5 rows:")
    print(df.head())
    
    print("\nDataset Information:")
    print(df.info())
    
    print("\nBasic Statistics:")
    print(df.describe())
    
    # Check for missing values
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    return df

# 2. Exploratory Data Analysis (EDA)
def perform_eda(df):
    """
    Perform exploratory data analysis on the stock data
    """
    # Create a figure with multiple subplots
    plt.figure(figsize=(18, 12))
    
    # 1. Plot closing price over time
    plt.subplot(3, 2, 1)
    plt.plot(df.index, df['Close'], label='Close Price', color='blue')
    plt.title('Stock Closing Price Over Time')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True)
    
    # 2. Plot trading volume over time
    plt.subplot(3, 2, 2)
    plt.bar(df.index, df['Volume'], color='purple', alpha=0.7)
    plt.title('Trading Volume Over Time')
    plt.xlabel('Date')
    plt.ylabel('Volume')
    plt.yscale('log')  # Log scale for better visualization
    plt.grid(True)
    
    # 3. Plot daily price range (High - Low)
    plt.subplot(3, 2, 3)
    df['Price_Range'] = df['High'] - df['Low']
    plt.plot(df.index, df['Price_Range'], color='green')
    plt.title('Daily Price Range (High - Low)')
    plt.xlabel('Date')
    plt.ylabel('Price Range ($)')
    plt.grid(True)
    
    # 4. Plot daily returns
    plt.subplot(3, 2, 4)
    df['Daily_Return'] = df['Close'].pct_change() * 100
    plt.plot(df.index, df['Daily_Return'], color='red')
    plt.title('Daily Returns (%)')
    plt.xlabel('Date')
    plt.ylabel('Return (%)')
    plt.grid(True)
    
    # 5. Distribution of daily returns
    plt.subplot(3, 2, 5)
    sns.histplot(df['Daily_Return'].dropna(), kde=True, bins=50, color='orange')
    plt.title('Distribution of Daily Returns')
    plt.xlabel('Daily Return (%)')
    plt.ylabel('Frequency')
    
    # 6. Correlation matrix
    plt.subplot(3, 2, 6)
    correlation_matrix = df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Matrix')
    
    plt.tight_layout()
    plt.savefig('eda_plots.png', dpi=300)
    plt.show()
    
    # Additional EDA - Checking for trends and seasonality
    plt.figure(figsize=(12, 8))
    
    # Weekly pattern
    df['Day_of_Week'] = df.index.dayofweek
    daily_avg = df.groupby('Day_of_Week')['Close'].mean()
    plt.subplot(2, 2, 1)
    sns.barplot(x=daily_avg.index, y=daily_avg.values)
    plt.title('Average Closing Price by Day of Week')
    plt.xlabel('Day of Week (0=Monday, 4=Friday)')
    plt.ylabel('Average Close Price')
    
    # Monthly pattern
    df['Month'] = df.index.month
    monthly_avg = df.groupby('Month')['Close'].mean()
    plt.subplot(2, 2, 2)
    sns.barplot(x=monthly_avg.index, y=monthly_avg.values)
    plt.title('Average Closing Price by Month')
    plt.xlabel('Month')
    plt.ylabel('Average Close Price')
    
    # Rolling statistics
    plt.subplot(2, 2, 3)
    df['Rolling_Mean_30'] = df['Close'].rolling(window=30).mean()
    df['Rolling_Std_30'] = df['Close'].rolling(window=30).std()
    plt.plot(df.index, df['Close'], label='Close Price', alpha=0.5)
    plt.plot(df.index, df['Rolling_Mean_30'], label='30-Day Rolling Mean', linewidth=2)
    plt.fill_between(df.index, 
                     df['Rolling_Mean_30'] - df['Rolling_Std_30'], 
                     df['Rolling_Mean_30'] + df['Rolling_Std_30'], 
                     color='gray', alpha=0.2)
    plt.title('30-Day Rolling Mean and Standard Deviation')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True)
    
    # Autocorrelation plot
    plt.subplot(2, 2, 4)
    from pandas.plotting import autocorrelation_plot
    autocorrelation_plot(df['Close'])
    plt.title('Autocorrelation Plot for Closing Price')
    
    plt.tight_layout()
    plt.savefig('additional_eda_plots.png', dpi=300)
    plt.show()
    
    return df

# 3. Feature Engineering
def engineer_features(df):
    """
    Create technical indicators and other features for the model
    """
    # Make a copy to avoid modifying the original dataframe
    features_df = df.copy()
    
    # Price-based features
    # Moving averages
    features_df['MA_5'] = features_df['Close'].rolling(window=5).mean()
    features_df['MA_10'] = features_df['Close'].rolling(window=10).mean()
    features_df['MA_20'] = features_df['Close'].rolling(window=20).mean()
    features_df['MA_50'] = features_df['Close'].rolling(window=50).mean()
    
    # Exponential moving averages
    features_df['EMA_5'] = features_df['Close'].ewm(span=5, adjust=False).mean()
    features_df['EMA_10'] = features_df['Close'].ewm(span=10, adjust=False).mean()
    features_df['EMA_20'] = features_df['Close'].ewm(span=20, adjust=False).mean()
    
    # Moving average convergence divergence (MACD)
    features_df['EMA_12'] = features_df['Close'].ewm(span=12, adjust=False).mean()
    features_df['EMA_26'] = features_df['Close'].ewm(span=26, adjust=False).mean()
    features_df['MACD'] = features_df['EMA_12'] - features_df['EMA_26']
    features_df['MACD_Signal'] = features_df['MACD'].ewm(span=9, adjust=False).mean()
    features_df['MACD_Hist'] = features_df['MACD'] - features_df['MACD_Signal']
    
    # Bollinger Bands
    features_df['BB_Middle'] = features_df['Close'].rolling(window=20).mean()
    features_df['BB_Std'] = features_df['Close'].rolling(window=20).std()
    features_df['BB_Upper'] = features_df['BB_Middle'] + 2 * features_df['BB_Std']
    features_df['BB_Lower'] = features_df['BB_Middle'] - 2 * features_df['BB_Std']
    features_df['BB_Width'] = (features_df['BB_Upper'] - features_df['BB_Lower']) / features_df['BB_Middle']
    
    # Relative Strength Index (RSI)
    delta = features_df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    features_df['RSI'] = 100 - (100 / (1 + rs))
    
    # Price rate of change
    features_df['ROC_5'] = features_df['Close'].pct_change(periods=5) * 100
    features_df['ROC_10'] = features_df['Close'].pct_change(periods=10) * 100
    features_df['ROC_20'] = features_df['Close'].pct_change(periods=20) * 100
    
    # Price momentum
    features_df['Momentum_5'] = features_df['Close'] - features_df['Close'].shift(5)
    features_df['Momentum_10'] = features_df['Close'] - features_df['Close'].shift(10)
    
    # Volatility features
    # Average True Range (ATR)
    high_low = features_df['High'] - features_df['Low']
    high_close = np.abs(features_df['High'] - features_df['Close'].shift())
    low_close = np.abs(features_df['Low'] - features_df['Close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    features_df['ATR_14'] = true_range.rolling(window=14).mean()
    
    # Historical volatility
    features_df['Volatility_5'] = features_df['Close'].pct_change().rolling(window=5).std() * np.sqrt(252)
    features_df['Volatility_10'] = features_df['Close'].pct_change().rolling(window=10).std() * np.sqrt(252)
    features_df['Volatility_20'] = features_df['Close'].pct_change().rolling(window=20).std() * np.sqrt(252)
    
    # Volume-based features
    features_df['Volume_MA_5'] = features_df['Volume'].rolling(window=5).mean()
    features_df['Volume_MA_10'] = features_df['Volume'].rolling(window=10).mean()
    features_df['Volume_Ratio'] = features_df['Volume'] / features_df['Volume_MA_5']
    
    # On-balance volume (OBV)
    features_df['OBV'] = np.nan
    obv_values = (features_df['Volume'].values[1:] * 
                ((features_df['Close'].values[1:] - 
                    features_df['Close'].values[:-1]) > 0).astype(int) * 2 - 1).cumsum()
    features_df.iloc[1:, features_df.columns.get_loc('OBV')] = obv_values
    features_df['OBV'].iloc[0] = 0  # Optional initialization
    
    # Volume-weighted average price (VWAP)
    features_df['Typical_Price'] = (features_df['High'] + features_df['Low'] + features_df['Close']) / 3
    features_df['TP_Volume'] = features_df['Typical_Price'] * features_df['Volume']
    features_df['TP_Volume_Cum'] = features_df['TP_Volume'].cumsum()
    features_df['Volume_Cum'] = features_df['Volume'].cumsum()
    features_df['VWAP'] = features_df['TP_Volume_Cum'] / features_df['Volume_Cum']
    
    # Calendar features
    features_df['Day_of_Week'] = features_df.index.dayofweek
    features_df['Month'] = features_df.index.month
    features_df['Quarter'] = features_df.index.quarter
    features_df['Year'] = features_df.index.year
    features_df['Day_of_Year'] = features_df.index.dayofyear
    
    # Target variable: Closing price 5 days into the future
    features_df['Target'] = features_df['Close'].shift(-5)
    
    features_df = features_df.dropna(subset=['Target'])
    # Then forward fill other NaNs
    features_df = features_df.fillna(method='ffill')
    # Fill any remaining NaNs with zeros
    features_df = features_df.fillna(0)
    
    # Show feature importance visualization using correlation with target
    plt.figure(figsize=(12, 10))
    correlations = features_df.corr()['Target'].sort_values(ascending=False)
    top_correlations = correlations.iloc[1:21]  # Top 20 correlations (excluding target itself)
    sns.barplot(x=top_correlations.values, y=top_correlations.index)
    plt.title('Top 20 Features Correlated with Target')
    plt.xlabel('Correlation with Target')
    plt.tight_layout()
    plt.savefig('feature_correlation.png', dpi=300)
    plt.show()
    
    print(f"Total features after engineering: {features_df.shape[1] - 1}")  # Exclude target column
    
    return features_df

# 4. Model Training and Evaluation
def train_and_evaluate_models(features_df):
    """
    Train and evaluate different models for stock price prediction
    """
    # Split the data: Use the last 20% of the data as the test set to maintain the time series nature
    train_size = int(len(features_df) * 0.8)
    train_data = features_df.iloc[:train_size]
    test_data = features_df.iloc[train_size:]
    
    print(f"Training data: {train_data.shape}, Test data: {test_data.shape}")
    
    # Define features and target
    feature_columns = [col for col in features_df.columns if col not in ['Target', 'Adj Close', 'TP_Volume_Cum', 'Volume_Cum']]
    X_train = train_data[feature_columns]
    y_train = train_data['Target']
    X_test = test_data[feature_columns]
    y_test = test_data['Target']
    
    # Normalize the features
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
        "XGBoost": xgb.XGBRegressor(n_estimators=100, random_state=42),
        "LightGBM": LGBMRegressor(n_estimators=100, random_state=42)
    }
    
    # Train and evaluate models
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Directional accuracy (whether the model correctly predicts the direction of price movement)
        actual_direction = np.sign(y_test - test_data['Close'])
        predicted_direction = np.sign(y_pred - test_data['Close'])
        directional_accuracy = np.mean(actual_direction == predicted_direction)
        
        results[name] = {
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'Directional Accuracy': directional_accuracy,
            'Model': model,
            'Predictions': y_pred
        }
        
        print(f"{name} Results:")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R2: {r2:.4f}")
        print(f"Directional Accuracy: {directional_accuracy:.4f}")
    
    # Visualize model comparison
    plt.figure(figsize=(15, 10))
    
    # Plot RMSE comparison
    plt.subplot(2, 2, 1)
    model_names = list(results.keys())
    rmse_values = [results[model]['RMSE'] for model in model_names]
    sns.barplot(x=model_names, y=rmse_values)
    plt.title('RMSE Comparison')
    plt.xticks(rotation=45)
    plt.ylabel('RMSE')
    
    # Plot MAE comparison
    plt.subplot(2, 2, 2)
    mae_values = [results[model]['MAE'] for model in model_names]
    sns.barplot(x=model_names, y=mae_values)
    plt.title('MAE Comparison')
    plt.xticks(rotation=45)
    plt.ylabel('MAE')
    
    # Plot R2 comparison
    plt.subplot(2, 2, 3)
    r2_values = [results[model]['R2'] for model in model_names]
    sns.barplot(x=model_names, y=r2_values)
    plt.title('R2 Comparison')
    plt.xticks(rotation=45)
    plt.ylabel('R2')
    
    # Plot Directional Accuracy comparison
    plt.subplot(2, 2, 4)
    da_values = [results[model]['Directional Accuracy'] for model in model_names]
    sns.barplot(x=model_names, y=da_values)
    plt.title('Directional Accuracy Comparison')
    plt.xticks(rotation=45)
    plt.ylabel('Directional Accuracy')
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300)
    plt.show()
    
    # Select the best model based on lowest RMSE and highest directional accuracy
    best_model_name = min(results, key=lambda x: results[x]['RMSE'])
    print(f"\nBest model based on RMSE: {best_model_name}")
    
    # Analyze the best model
    best_model = results[best_model_name]['Model']
    best_predictions = results[best_model_name]['Predictions']
    
    # Visualize actual vs predicted prices
    plt.figure(figsize=(12, 6))
    plt.plot(test_data.index, y_test, label='Actual Price', color='blue')
    plt.plot(test_data.index, best_predictions, label='Predicted Price', color='red', linestyle='--')
    plt.title(f'Actual vs Predicted Prices - {best_model_name}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.savefig('actual_vs_predicted.png', dpi=300)
    plt.show()
    
    # Feature importance analysis
    if hasattr(best_model, 'feature_importances_'):
        # For tree-based models
        feature_importance = pd.DataFrame({
            'Feature': feature_columns,
            'Importance': best_model.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=feature_importance.head(20))
        plt.title(f'Top 20 Feature Importance - {best_model_name}')
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300)
        plt.show()
        
        print("\nTop 10 Important Features:")
        print(feature_importance.head(10))
    
    # Simulation of a trading strategy using the predictions
    simulate_trading_strategy(test_data, best_predictions, best_model_name)
    
    return best_model, results, feature_columns, scaler

# 5. Trading Strategy Simulation
def simulate_trading_strategy(test_data, predictions, model_name):
    """
    Simulate a simple trading strategy based on model predictions
    """
    # Create a dataframe for simulation
    simulation_df = pd.DataFrame({
        'Date': test_data.index,
        'Actual_Close': test_data['Close'],
        'Predicted_Next_5d': predictions
    })
    
    # Trading logic:
    # Buy if predicted price is higher than current price by at least 1%
    # Sell if predicted price is lower than current price by at least 1%
    # Hold otherwise
    
    simulation_df['Position'] = np.where(
        simulation_df['Predicted_Next_5d'] > simulation_df['Actual_Close'] * 1.01, 1,  # Buy
        np.where(simulation_df['Predicted_Next_5d'] < simulation_df['Actual_Close'] * 0.99, -1,  # Sell
                0)  # Hold
    )
    
    # Calculate daily returns of the stock
    simulation_df['Stock_Return'] = simulation_df['Actual_Close'].pct_change()
    
    # Calculate strategy returns:
    # If position is 1 (long), gain the stock return
    # If position is -1 (short), gain the negative stock return
    # If position is 0 (hold), no return
    simulation_df['Strategy_Return'] = simulation_df['Position'].shift(1) * simulation_df['Stock_Return']
    
    # Calculate cumulative returns
    simulation_df['Cum_Stock_Return'] = (1 + simulation_df['Stock_Return']).cumprod() - 1
    simulation_df['Cum_Strategy_Return'] = (1 + simulation_df['Strategy_Return']).cumprod() - 1
    
    # Calculate additional metrics
    total_trades = (simulation_df['Position'].diff() != 0).sum()
    winning_trades = (simulation_df['Strategy_Return'] > 0).sum()
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    
    # Calculate maximum drawdown
    strategy_equity = (1 + simulation_df['Strategy_Return']).cumprod()
    running_max = strategy_equity.cummax()
    drawdown = (strategy_equity - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Calculate Sharpe ratio (assuming risk-free rate of 0)
    sharpe_ratio = simulation_df['Strategy_Return'].mean() / simulation_df['Strategy_Return'].std() * np.sqrt(252)
    
    # Visualize the strategy performance
    plt.figure(figsize=(12, 8))
    
    # Plot cumulative returns
    plt.subplot(2, 1, 1)
    plt.plot(simulation_df['Date'], simulation_df['Cum_Stock_Return'], label='Buy & Hold', color='blue')
    plt.plot(simulation_df['Date'], simulation_df['Cum_Strategy_Return'], label='Strategy', color='green')
    plt.title(f'Trading Strategy Performance - {model_name}')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True)
    
    # Plot positions over time
    plt.subplot(2, 1, 2)
    plt.plot(simulation_df['Date'], simulation_df['Position'], color='black')
    plt.title('Trading Positions Over Time')
    plt.xlabel('Date')
    plt.ylabel('Position (1=Long, -1=Short, 0=Hold)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('trading_strategy.png', dpi=300)
    plt.show()
    
    # Print strategy performance metrics
    print("\nTrading Strategy Performance Metrics:")
    print(f"Total Trades: {total_trades}")
    print(f"Win Rate: {win_rate:.4f}")
    print(f"Cumulative Return: {simulation_df['Cum_Strategy_Return'].iloc[-1]:.4f}")
    print(f"Buy & Hold Return: {simulation_df['Cum_Stock_Return'].iloc[-1]:.4f}")
    print(f"Maximum Drawdown: {max_drawdown:.4f}")
    print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
    
    return simulation_df

# 6. Make Future Predictions
def make_future_predictions(best_model, feature_columns, scaler, latest_data):
    """
    Make predictions for the next 5 days with different values for each day
    """
    # Clean NaT values from the index and use the clean data
    clean_data = latest_data[latest_data.index.notnull()].copy()
    
    # Initialize an empty list to store predictions
    future_prices = []
    
    # Get the last date from the data
    last_date = clean_data.index[-1]
    
    # Create business days for the future
    future_dates = pd.date_range(start=last_date, periods=6, freq='B')[1:]  # Skip current day
    
    # Create a copy of the latest data to simulate future days
    future_data = clean_data.iloc[-20:].copy()  # Taking last 20 rows to have enough data for feature calculation
    
    # For each future day
    for i in range(5):
        # Get features for the current state
        current_features = future_data[feature_columns].iloc[-1:].values
        
        # Scale the features
        current_features_scaled = scaler.transform(current_features)
        
        # Make prediction for the next day
        next_day_price = best_model.predict(current_features_scaled)[0]
        
        # Store the prediction
        future_prices.append(next_day_price)
        
        # Create a new row with the predicted price
        new_row = future_data.iloc[-1:].copy()
        new_row.index = [future_dates[i]]
        new_row['Close'] = next_day_price
        new_row['Open'] = next_day_price  # Simplified assumption
        new_row['High'] = next_day_price * 1.005  # Simplified assumption: high is 0.5% above close
        new_row['Low'] = next_day_price * 0.995   # Simplified assumption: low is 0.5% below close
        new_row['Adj Close'] = next_day_price
        new_row['Volume'] = future_data['Volume'].mean()  # Use average volume
        
        # Update features for the new row (recalculate technical indicators)
        # This is a simplified approach - in practice, you would recalculate all features
        for ma in [5, 10, 20, 50]:
            ma_col = f'MA_{ma}'
            if ma_col in feature_columns:
                # Fixed: Use pd.concat instead of append
                temp_series = pd.concat([future_data['Close'].iloc[-ma:], pd.Series([next_day_price])])
                new_row[ma_col] = temp_series.mean() if len(future_data) >= ma else future_data['Close'].mean()
        
        # Append the new row to future_data for the next iteration
        future_data = pd.concat([future_data, new_row])
    
    # Get current price
    current_price = clean_data['Close'].iloc[-1]
    
    # Print predictions
    print("\nFuture Price Predictions:")
    print(f"Current Price: ${current_price:.2f}")
    for i, (date, price) in enumerate(zip(future_dates, future_prices)):
        expected_return = (price - current_price) / current_price * 100
        print(f"Day {i+1} ({date.strftime('%Y-%m-%d')}): ${price:.2f} (Expected Return: {expected_return:.2f}%)")
    
    # Create a DataFrame for the predictions
    predictions_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted_Close': future_prices
    })
    
    # Save predictions to CSV
    predictions_df.to_csv('future_predictions.csv', index=False)
    print("Future predictions saved to 'future_predictions.csv'")
    
    # Visualize the predictions
    plt.figure(figsize=(12, 6))
    
    # Plot historical data
    plt.plot(clean_data.index[-30:], clean_data['Close'].iloc[-30:], label='Historical Close', color='blue')
    
    # Plot predictions
    plt.plot(future_dates, future_prices, label='Predicted Close', color='red', linestyle='--', marker='o')
    
    # Add labels and title
    plt.title('Stock Price Prediction for Next 5 Business Days')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('future_predictions_plot.png', dpi=300)
    plt.show()
    
    return predictions_df

# 8. Main Function
def main(file_path):
    """
    Main function to orchestrate the entire workflow
    """
    # Step 1: Load and explore the data
    print("Step 1: Loading and exploring the data...")
    df = load_and_explore_data(file_path)
    
    # Step 2: Perform EDA
    print("\nStep 2: Performing Exploratory Data Analysis...")
    df = perform_eda(df)
    
    # Step 3: Engineer features
    print("\nStep 3: Engineering features...")
    features_df = engineer_features(df)
    
    # Step 4: Train and evaluate models
    print("\nStep 4: Training and evaluating models...")
    best_model, results, feature_columns, scaler = train_and_evaluate_models(features_df)
    
    # Step 5: Make future predictions
    print("\nStep 5: Making future predictions...")
    future_predictions = make_future_predictions(best_model, feature_columns, scaler, features_df)
    
    # End
    print("\nStock price prediction workflow completed successfully!")

# Execute the main function if this script is run directly
if __name__ == "__main__":
    file_path = "./question4-stock-data.csv"  # Change this to your actual file path
    main(file_path)