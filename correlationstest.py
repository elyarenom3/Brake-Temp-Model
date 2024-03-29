import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import h5py

# base_dir should be only thing you need to change to run locally
base_dir = "/Users/adminelya/Desktop/Mclaren/data"


def load_race_data(base_directory, race_names):
    all_data = []
    for race in race_names:
        file_path = os.path.join(base_directory, f"{race}.hdf5")
        with h5py.File(file_path, 'r') as f:
            data = pd.DataFrame(f['data'][:], columns=f.attrs['channel_names'])
            data['Race'] = race
            all_data.append(data)
    return pd.concat(all_data, ignore_index=True)


race_names = ['Austria', 'Bahrain', 'Hungary', 'Portugal', 'Spain', 'Turkey']
df = load_race_data(base_dir, race_names)

# df.drop(columns=["TTrack", "TAir"], inplace=True)

# Calculate deltas and averages

# df['deceleration'] = np.where(df['deltaSpeed'] < 0, df['deltaSpeed'], np.nan)
df['deltaTBrakeL'] = df['TBrakeL'].diff()
df['deltaTBrakeR'] = df['TBrakeR'].diff()
df['deltaSpeed'] = df['vCar'].diff()
df = df[df['deltaSpeed'] < -2]
print(df.describe())

# Clean up the DataFrame by dropping the individual brake temperature columns
# df.drop(columns=["deltaTBrakeR", "deltaTBrakeL", "TBrakeR", "TBrakeL"], inplace=True)
# Select only numeric columns for correlation calculation
numeric_df = df.select_dtypes(include=[np.number])

plt.figure(figsize=(10, 8))
corr = numeric_df.corr()  # Calculate correlation on numeric columns only
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()
