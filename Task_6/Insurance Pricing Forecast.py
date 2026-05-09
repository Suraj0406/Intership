# ==========================================
# Insurance Pricing Forecast (Correct Code)
# ==========================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor

# Load Dataset
data = pd.read_csv(r"F:\python\Task_6\insurance.csv")

print("Dataset Preview")
print(data.head())


# ==========================================
# Encode Categorical Data
# ==========================================

label_encoder = LabelEncoder()

data["sex"] = label_encoder.fit_transform(data["sex"])
data["smoker"] = label_encoder.fit_transform(data["smoker"])
data["region"] = label_encoder.fit_transform(data["region"])


# ==========================================
# Define Features and Target
# ==========================================

X = data.drop("expenses", axis=1)
y = data["expenses"]


# ==========================================
# Train Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)


# ==========================================
# Train Model
# ==========================================

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)


# ==========================================
# Predictions
# ==========================================

predictions = model.predict(X_test)


# ==========================================
# Evaluation
# ==========================================

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

print("\nModel Performance")
print("MAE :", mae)
print("RMSE:", rmse)


# ==========================================
# Example Prediction
# ==========================================

sample = [[30, 1, 28.5, 1, 0, 2]]

prediction = model.predict(sample)

print("\nPredicted Insurance Cost:", prediction[0])