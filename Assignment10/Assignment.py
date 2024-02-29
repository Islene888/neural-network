import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Assuming 'diabetes_data' is your dataset
diabetes_data = pd.read_csv('path_to_your_file/diabetes.csv')  # Replace 'path_to_your_file/' with the actual file path

# Separate features and target
X = diabetes_data.drop('Outcome', axis=1)
y = diabetes_data['Outcome']

# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# Fit the scaler to the features and transform them
X_normalized = scaler.fit_transform(X)

# Convert the normalized features back to a DataFrame for easier handling
X_normalized_df = pd.DataFrame(X_normalized, columns=X.columns)

# Now, X_normalized_df contains the normalized features
