import pandas as pd
from joblib import load
import matplotlib.pyplot as plt


def plot_feature_importance(model_path, feature_names, model_name):
    model = load(model_path)
    importance = model.feature_importances_
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
    importance_df.sort_values(by='Importance', ascending=True, inplace=True)

    print(f'Most Important Feature for {model_name}:', importance_df.iloc[-1]['Feature'], 'with an importance of',
          importance_df.iloc[-1]['Importance'])

    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['Feature'], importance_df['Importance'])
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title(f'Feature Importance for {model_name}')
    plt.show()


feature_names = ['vCar', 'TAir', 'TTrack','deltaSpeed']

model_paths = {
    'Model R Greater Than or Equal to 2': 'greater_than_or_equal_2_deltaTBrakeR_model.joblib',
    'Model L Greater Than or Equal to 2': 'greater_than_or_equal_2_deltaTBrakeL_model.joblib',
    'Model R Less Than -2': 'less_than_minus_2_deltaTBrakeR_model.joblib',
    'Model L Less Than -2': 'less_than_minus_2_deltaTBrakeL_model.joblib'
}

for model_name, model_path in model_paths.items():
    plot_feature_importance(model_path, feature_names, model_name)