from datetime import datetime, timedelta
import yfinance as yf
import matplotlib.pyplot as plt
import joblib
import json
from utils.preprocess import construct_date, date_number

# Opening and loading config JSON file
config_file = open('config.json')
data = json.load(config_file)

# Unpacking variables
weeks, share, weeks_data, models_folder = data['weeks'], data['share'], data['weeks_data'], data["models_folder"]

# Setting date range for dataset download
end_date = datetime.today()
start_date = end_date - timedelta(weeks=weeks)

# Downloading the data
data = yf.download(share, start=start_date,
                        end=end_date)

# Preprocessing the data
df = data.drop(['Adj Close'], axis=1)
df['Date'] = df.index
df['day'] = df['Date'].apply(lambda r:r.weekday())
df['date_number'] = df['Date'].apply(lambda r:date_number(r))

days = list(df['Date'].apply(lambda r: construct_date(r, weeks= weeks)))
df = df.drop(['Date'], axis=1)


print("Number of rows in data: ", len(df.index))
print(df.head())

# Assigning data to variable
X = df.iloc[:, :7].values

file_location = f"{models_folder}/{share}/random_forest_model_{share}_{weeks}.pkl"
loaded_rf_model = joblib.load(file_location)
pred =  loaded_rf_model.predict(X)

# Plotting the predicted values in green
# days = list(range(1, len(pred)+1))
plt.plot(days, pred, color='green', label='Predicted')

# Adding labels and title
plt.xlabel('Days')
plt.ylabel(f'Stock Price {share}')
plt.xticks(rotation=45)
plt.xticks(range(0, len(days), 5), days[::5])
plt.title('Predicted')

# Adding a legend
plt.legend()

# Show the plot
plt.show()