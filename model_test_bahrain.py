import pandas as pd
import h5py
from joblib import load
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Using Bahrain as our 'new' data
file_path = "/Users/adminelya/Desktop/Mclaren/data/Bahrain.hdf5"


def preprocess_new_data(file_path):
    with h5py.File(file_path, "r") as file:
        data = pd.DataFrame(file['data'][:], columns=file.attrs['channel_names'])
    data['deltaSpeed'] = data['vCar'].diff()
    data['deltaTBrakeL'] = data['TBrakeL'].diff()
    data['deltaTBrakeR'] = data['TBrakeR'].diff()
    return data


models = {
    'model_R_less': load('less_than_or_equal_to_minus_2_deltaTBrakeR_model.joblib'),
    'model_L_less': load('less_than_or_equal_to_minus_2_deltaTBrakeL_model.joblib'),
    'model_R_greater': load('more_than_minus_2_deltaTBrakeR_model.joblib'),
    'model_L_greater': load('more_than_minus_2_deltaTBrakeL_model.joblib'),
}

new_data = preprocess_new_data(file_path)


# Initial brake temperatures from the first data point
initial_brake_temps = {
    'TBrakeR': new_data['TBrakeR'].iloc[0],
    'TBrakeL': new_data['TBrakeL'].iloc[0]
}


def preprocess_and_predict(new_data, models, initial_brake_temps, n_predictions):
    predicted_TBrakeR = [initial_brake_temps['TBrakeR']]
    predicted_TBrakeL = [initial_brake_temps['TBrakeL']]
    features = ['deltaSpeed', 'vCar', 'TAir', 'TTrack']

    for i in range(1, n_predictions):
        if i < len(new_data):
            current_features = new_data.iloc[i][features].values.reshape(1, -1)
            delta_speed = new_data.iloc[i]['deltaSpeed']
        else:
            current_features = new_data.iloc[-1][features].values.reshape(1, -1)
            delta_speed = new_data.iloc[-1]['deltaSpeed']

        if delta_speed >= -2:
            delta_R = models['model_R_greater'].predict(current_features)[0]
            delta_L = models['model_L_greater'].predict(current_features)[0]
        else:
            delta_R = models['model_R_less'].predict(current_features)[0]
            delta_L = models['model_L_less'].predict(current_features)[0]

        predicted_TBrakeR.append(predicted_TBrakeR[-1] + delta_R)
        predicted_TBrakeL.append(predicted_TBrakeL[-1] + delta_L)

    return pd.Series(predicted_TBrakeR, name='Predicted_TBrakeR'), pd.Series(predicted_TBrakeL, name='Predicted_TBrakeL')


n_predictions = len(new_data)
predicted_TBrakeR_series, predicted_TBrakeL_series = preprocess_and_predict(new_data, models, initial_brake_temps, n_predictions)

predicted_temps = pd.DataFrame({
    'TBrakeR': predicted_TBrakeR_series,
    'TBrakeL': predicted_TBrakeL_series
})

# Adjust actual temperatures to match the number of predictions for fair comparison
actual_temps_matched = new_data[['TBrakeR', 'TBrakeL']].head(n_predictions + 1)


def evaluate_model(actual_temps, predicted_temps):
    mse_r = mean_squared_error(actual_temps['TBrakeR'], predicted_temps['TBrakeR'])
    mse_l = mean_squared_error(actual_temps['TBrakeL'], predicted_temps['TBrakeL'])
    print(f'MSE for Right Brake Temperature: {mse_r}')
    print(f'MSE for Left Brake Temperature: {mse_l}')


evaluate_model(actual_temps_matched, predicted_temps)

mse_values_r = []
mse_values_l = []

# Can change this to however many predictions into the future we want
predictions_eval = 200
for n in range(1, predictions_eval):
    predicted_TBrakeR_series, predicted_TBrakeL_series = preprocess_and_predict(new_data, models, initial_brake_temps,
                                                                                n)

    predicted_temps = pd.DataFrame({
        'TBrakeR': predicted_TBrakeR_series,
        'TBrakeL': predicted_TBrakeL_series
    })

    # Adjust actual temperatures to match the number of predictions for fair comparison
    actual_temps_matched = new_data[['TBrakeR', 'TBrakeL']].head(n + 1)

    mse_r = mean_squared_error(actual_temps_matched['TBrakeR'], predicted_temps['TBrakeR'])
    mse_l = mean_squared_error(actual_temps_matched['TBrakeL'], predicted_temps['TBrakeL'])

    mse_values_r.append(mse_r)
    mse_values_l.append(mse_l)

# Plotting the MSE values
plt.figure(figsize=(10, 6))
plt.plot(range(1, predictions_eval), mse_values_r, label='MSE Right Brake', color='r')
plt.plot(range(1, predictions_eval), mse_values_l, label='MSE Left Brake', color='b')
plt.xlabel('Number of Predictions')
plt.ylabel('Mean Squared Error')
plt.title('MSE of Brake Temperatures as Number of Predictions Increases (Bahrain)')
plt.legend()
plt.grid(True)
plt.show()
