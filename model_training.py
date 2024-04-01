import os
import numpy as np
import pandas as pd
import h5py
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from joblib import dump

base_dir = "/Users/adminelya/Desktop/Mclaren/data"


def load_race_data(base_directory, race_names):
    all_data = []
    for race in race_names:
        file_path = os.path.join(base_directory, f"{race}.hdf5")
        with h5py.File(file_path, "r") as file:
            data = pd.DataFrame(file['data'][:], columns=file.attrs['channel_names'])
            data['Race'] = race
            all_data.append(data)
    return pd.concat(all_data, ignore_index=True)


def replace_outliers_with_mean(column):
    col_mean = column.mean()
    col_std = column.std()
    z_scores = (column - col_mean) / col_std
    outliers = np.abs(z_scores) > 15
    column_cleaned = column.where(~outliers, col_mean)
    return column_cleaned


def preprocess_data(df):
    # Would clean data more thoroughly with more time
    df['deltaSpeed'] = df['vCar'].diff()
    df['deltaTBrakeL'] = df['TBrakeL'].diff()
    df['deltaTBrakeR'] = df['TBrakeR'].diff()
    df['deltaTBrakeL'] = replace_outliers_with_mean(df['deltaTBrakeL'])
    df['deltaTBrakeR'] = replace_outliers_with_mean(df['deltaTBrakeR'])
    return df


race_names = ['Austria', 'Bahrain', 'Hungary', 'Portugal', 'Spain', 'Turkey']
df = load_race_data(base_dir, race_names)
df = preprocess_data(df)


def time_based_split(X, y, train_size_ratio=0.7):
    split_index = int(len(X) * train_size_ratio)
    return X.iloc[:split_index], X.iloc[split_index:], y.iloc[:split_index], y.iloc[split_index:]


def train_and_evaluate(X_train, X_test, y_train, y_test, param_grid, cv_splits):
    grid_search = GridSearchCV(
        estimator=XGBRegressor(objective='reg:squarederror'),
        param_grid=param_grid,
        cv=TimeSeriesSplit(n_splits=cv_splits),
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=2,
    )
    grid_search.fit(X_train, y_train)
    y_pred = grid_search.best_estimator_.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return grid_search.best_estimator_, mse, r2, y_pred


param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}
cv_splits = 5

condition_below = df['deltaSpeed'] <= -2
condition_above = df['deltaSpeed'] > -2

for condition, name in zip([condition_below, condition_above], ['less_than_or_equal_to_minus_2', 'more_than_minus_2']):
    condition_df = df[condition]
    features = condition_df.drop(columns=["TBrakeR", "TBrakeL", "deltaTBrakeR", "deltaTBrakeL", "Race"])
    targets = condition_df[["deltaTBrakeR", "deltaTBrakeL"]]
    for target in targets.columns:
        X_train, X_test, y_train, y_test = time_based_split(features, condition_df[target])
        best_estimator, mse, r2, y_pred = train_and_evaluate(X_train, X_test, y_train, y_test, param_grid, cv_splits)
        model_key = f"{name}_{target}"

        plt.figure(figsize=(10, 5))
        plt.plot(y_test.values, label='Actual', alpha=0.7)
        plt.plot(y_pred, label='Predicted', linestyle='--')
        plt.title(f'Predicted vs Actual - {model_key}')
        plt.legend()
        plt.show()

        print(f"{model_key} - MSE: {mse}, R2: {r2}")
        dump(best_estimator, f'{model_key}_model.joblib')
