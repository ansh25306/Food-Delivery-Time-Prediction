import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Dataset
data = {
    'distance': [2, 5, 3, 8, 6],
    'prep_time': [10, 15, 12, 20, 18],
    'traffic': [1, 2, 1, 3, 2],
    'delivery_time': [20, 35, 25, 50, 40]
}

df = pd.DataFrame(data)

# Features & target
X = df[['distance', 'prep_time', 'traffic']]
y = df['delivery_time']

# Model
model = LinearRegression()
model.fit(X, y)

# Prediction
prediction = model.predict([[4, 12, 2]])
print("Predicted Delivery Time:", prediction)
