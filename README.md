# Food-Delivery-Time-Prediction
 Food Delivery Time Prediction using Linear Regression

##  Project Overview
This project uses **Linear Regression (Machine Learning)** to predict the delivery time of food orders based on factors like distance, preparation time, and traffic conditions.

---

##  Objective
- Predict delivery time accurately  
- Understand relationship between variables  
- Apply ML in real-world scenario  

---

##  Concept Used

Linear Regression Model:

y = b0 + b1x1 + b2x2 + b3x3

Where:
- y = Delivery Time  
- x1 = Distance  
- x2 = Preparation Time  
- x3 = Traffic Level  


##  Technologies Used
- Python  
- NumPy  
- Pandas  
- Matplotlib  
- Scikit-learn  

---

## Implementation

```python
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

X = df[['distance', 'prep_time', 'traffic']]
y = df['delivery_time']

model = LinearRegression()
model.fit(X, y)

prediction = model.predict([[4, 12, 2]])
print("Predicted Delivery Time:", prediction)





 ## Output

 Predicted Delivery Time ≈ 30 minutes

 ## Applications
 1. Food delivery platforms
 2. Logistics and courier services
 3. Route optimization systems
 ## Advantages
 1. Simple and easy to implement
 2. Fast computation
 3. Good for linear data
