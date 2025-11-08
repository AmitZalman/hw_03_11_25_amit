import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error , r2_score


#Create data:

#X = [Years of Experience, Education (Years), Weekly Working Hours, Management Experience (Years)]
X = np.array([
    [2, 15, 40, 0],
    [5, 16, 45, 1],
    [3, 18, 50, 0],
    [10, 18, 55, 5],
    [7, 17, 45, 3],
    [1, 14, 40, 1],
    [8, 16, 46, 4],
    [4, 15, 40, 1],
    [6, 17, 42, 2],
    [12, 19, 55, 8]
])
# y = Monthly salary thousends
Y_observetion = np.array([15, 25, 18, 45, 35, 22, 38, 22, 30, 60])


# Fit the model
model = LinearRegression()
model.fit(X , Y_observetion)

#Get coef , Intercept
print(f"Intercept (β₀): {model.intercept_:.2f}")
print(f"Coefficients (β₁, β₂, β₃, β₄): {model.coef_}")

#r2 - (sse , sst)
y_pred_hat = model.predict(X)
y_mean_bar = np.mean(Y_observetion)

sse = np.sum((Y_observetion - y_pred_hat)**2)
sst = np.sum((Y_observetion - y_mean_bar)**2)
r2 = 1 - ( sse / sst)
print(f"r2 = {r2}")

#r2 - r2_score
print("r2_score = " , r2_score(Y_observetion , y_pred))

#prediction
new_employee = np.array([[6, 16, 44, 2]])
predicted_salary = model.predict(new_employee)

print(f"Predicted Monthly Salary: {predicted_salary[0]:.2f} thousand NIS")
