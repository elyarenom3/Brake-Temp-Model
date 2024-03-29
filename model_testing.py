import pandas as pd
import h5py
from joblib import load
from sklearn.metrics import mean_squared_error
'''
This file is for the purposes of testing my model.
'''

new_file_path = "/Path/to/new/unseen/data"  # Replace with the actual file path
# Load the trained models
model_R_less = load('model_less_than_minus_2_deltaTBrakeR_model.joblib')
model_L_less = load('model_less_than_minus_2_deltaTBrakeL_model.joblib')
model_R_greater = load('model_greater_than_or_equal_2_deltaTBrakeR_model.joblib')
model_L_greater = load('model_greater_than_or_equal_2_deltaTBrakeL_model.joblib')
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
    # Unpack the models for clarity
    model_R_greater = models['model_greater_than_or_equal_2_deltaTBrakeR']
    model_L_greater = models['model_greater_than_or_equal_2_deltaTBrakeL']
    model_R_less = models['model_less_than_minus_2_deltaTBrakeR']
    model_L_less = models['model_less_than_minus_2_deltaTBrakeL']

    # Initialize lists to store brake temperature predictions
    predicted_TBrakeR = [initial_brake_temps['TBrakeR']]
    predicted_TBrakeL = [initial_brake_temps['TBrakeL']]

    for i in range(1, len(new_data)):
        current_features = new_data.iloc[i][['vCar', 'TAir', 'TTrack']].values.reshape(1, -1)
        delta_speed = new_data.iloc[i - 1]['deltaSpeed']  # Using i-1 since deltaSpeed is based on the previous step

        if delta_speed >= 2:
            delta_R = model_R_greater.predict(current_features)[0]  # Assuming predict returns an array, take the first element
            delta_L = model_L_greater.predict(current_features)[0]
        else:
            delta_R = model_R_less.predict(current_features)[0]
            delta_L = model_L_less.predict(current_features)[0]

        # Add the predicted delta to the last known brake temperature and append to the list
        predicted_TBrakeR.append(predicted_TBrakeR[-1] + delta_R)
        predicted_TBrakeL.append(predicted_TBrakeL[-1] + delta_L)

    # Convert the lists of predictions into Pandas Series for easier handling
    predicted_TBrakeR_series = pd.Series(predicted_TBrakeR, name='Predicted_TBrakeR')
    predicted_TBrakeL_series = pd.Series(predicted_TBrakeL, name='Predicted_TBrakeL')

    return predicted_TBrakeR_series, predicted_TBrakeL_series


def evaluate_model(actual_temps, predicted_temps):
    mse_r = mean_squared_error(actual_temps['TBrakeR'], predicted_temps['TBrakeR'])
    mse_l = mean_squared_error(actual_temps['TBrakeL'], predicted_temps['TBrakeL'])
    print(f'MSE for Right Brake Temperature: {mse_r}')
    print(f'MSE for Left Brake Temperature: {mse_l}')

