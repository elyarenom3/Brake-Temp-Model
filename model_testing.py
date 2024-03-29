import pandas as pd
import numpy as np
import h5py
from joblib import load
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

new_file_path = "/Path/to/new/unseen/data"  # Replace with the actual file path
# Load the trained models
model_less_than_minus_2_deltaTBrakeR = load('less_than_minus_2_deltaTBrakeR_model.joblib')
model_less_than_minus_2_deltaTBrakeL = load('less_than_minus_2_deltaTBrakeL_model.joblib')
model_greater_than_or_equal_2_deltaTBrakeR = load('greater_than_or_equal_2_deltaTBrakeR_model.joblib')
model_greater_than_or_equal_2_deltaTBrakeL = load('greater_than_or_equal_2_deltaTBrakeL_model.joblib')

# Function to preprocess new data similar to the training data
def preprocess_new_data(file_path):
    with h5py.File(file_path, "r") as file:
        data = pd.DataFrame(file['data'][:], columns=file.attrs['channel_names'])
    data['deltaSpeed'] = data['vCar'].diff()
    data['deltaTBrakeL'] = data['TBrakeL'].diff()
    data['deltaTBrakeR'] = data['TBrakeR'].diff()
    return data

new_data = preprocess_new_data(new_file_path)


def preprocess_and_predict(new_data, models, initial_brake_temps):
    # Initial brake temperatures
    brake_temps = {'TBrakeR': [initial_brake_temps['TBrakeR']], 'TBrakeL': [initial_brake_temps['TBrakeL']]}

    for i in range(1, len(new_data)):
        new_data.loc[i, 'deltaSpeed'] = new_data.loc[i, 'vCar'] - new_data.loc[i - 1, 'vCar']

        # Determine which model to use based on deltaSpeed
        if new_data.loc[i, 'deltaSpeed'] >= 2:
            model_key_r = 'deltaTBrakeR_gte_2'
            model_key_l = 'deltaTBrakeL_gte_2'
        else:
            # If deltaSpeed condition for other models (e.g., < -2) needs to be handled, add here
            continue

        # Extract the row as a single sample for prediction
        sample = new_data.iloc[i].to_frame().T.drop(columns=['deltaSpeed'])

        # Predict the deltas
        delta_r = models[model_key_r].predict(sample)[0]
        delta_l = models[model_key_l].predict(sample)[0]

        # Update the brake temperatures by adding the predicted deltas to the last known temperature
        brake_temps['TBrakeR'].append(brake_temps['TBrakeR'][-1] + delta_r)
        brake_temps['TBrakeL'].append(brake_temps['TBrakeL'][-1] + delta_l)

    # Convert lists to series for easier handling
    brake_temps['TBrakeR'] = pd.Series(brake_temps['TBrakeR'])
    brake_temps['TBrakeL'] = pd.Series(brake_temps['TBrakeL'])

    return brake_temps

# Example usage:
# Load your new_data DataFrame here
# initial_brake_temps = {'TBrakeR': starting_value_for_TBrakeR, 'TBrakeL': starting_value_for_TBrakeL}
# predicted_brake_temps = preprocess_and_predict(new_data, models, initial_brake_temps)
