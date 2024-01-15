from datetime import datetime, timedelta
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from utils.preprocess import roll_value, find_index_position, date_number
import joblib
import pandas as pd
import json
import os


# Opening and loading config JSON file
config_file = open('config.json')
data = json.load(config_file)

# Unpacking variables
weeks, share, weeks_data, models_folder = data['weeks'], data['share'], data['weeks_data'], data["models_folder"]

# Setting date range for dataset download
end_date = datetime.today()
start_date = end_date - timedelta(weeks=weeks_data)

# Downloading data from yahoo finance
data = yf.download(share, start=start_date,
                        end=end_date)[::-1]

print(f"Total number of rows: {len(data.index)}")


# Preprocessing the data
df = data.drop(['Adj Close'], axis=1)
df['Date'] = df.index
df['day'] = df['Date'].apply(lambda r:r.weekday())
df['date_number'] = df['Date'].apply(lambda r:date_number(r))
df['Target'] = df['Date'].apply(lambda r: roll_value(r, df, weeks=weeks))

print(df.head())
index_position = find_index_position(df)
train_df = df.iloc[index_position:, :]

print(train_df.head())

train_df.fillna(train_df.mean(), inplace=True)
train_df = train_df.drop(['Date'], axis=1)


# X contains the features, and y contains the target variable
X = train_df.iloc[:, :7].values
y = train_df.iloc[:, -1].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.02, random_state=42)

regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X, y)

pred = regressor.predict(X_test)
file_name = f'random_forest_model_{share}_{weeks}.pkl'
folder = f"{models_folder}/{share}"

os.makedirs(folder, exist_ok=True)
model_file_path = os.path.join(folder, file_name)

with open(model_file_path, 'wb') as model_file:
    joblib.dump(regressor, model_file)


days  = list(range(1, len(y_test)+1))  # Replace with your actual days data

# Plotting the actual values in red
plt.plot(days, y_test, color='red', label='Actual')

# Plotting the predicted values in green
plt.plot(days, pred, color='green', label='Predicted')

# Adding labels and title
plt.xlabel('Days')
plt.ylabel(f'Stock Price {share}')
plt.title('Actual vs. Predicted')

# Adding a legend
plt.legend()

# Show the plot
plt.show()