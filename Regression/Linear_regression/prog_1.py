# from urllib.request import urlretrieve
# import pandas as pd
# medical_charges_url = 'https://raw.githubusercontent.com/JovianML/opendatasets/master/data/medical-charges.csv'
# urlretrieve(medical_charges_url, 'medical.csv')
# csv = pd.read_csv("medical.csv")
# medical_df = pd.DataFrame(csv)
# print(medical_df)
# '''The dataset contains 1338 rows and 7 columns. Each row of the dataset contains information about one customer. 

# Our objective is to find a way to estimate the value in the "charges" column using the values in the other columns. If we can do so for the historical data, then we should able to estimate charges for new customers too, simply by asking for information like their age, sex, BMI, no. of children, smoking habits and region.Let's check the data type for each column.'''

# medical_df.info()
# medical_df.describe() # calculates the deviation and central item of the dataset

# import jovian

# jovian.commit(
#     message="Commit from VS Code",
#     files=["script.py", "dataset.csv"]
# )
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = {
    "x": np.array([1, 2, 4, 6, 8]),
    "y": np.array([3, 4, 8, 10, 15])
}

df = pd.DataFrame(data)
X = df[["x"]]   # Feature (must be 2D for sklearn)
Y = df["y"]
model = LinearRegression()
model.fit(X, Y)
y_pred = model.predict(X)

#new_value_prediction using previous model
x_new = pd.DataFrame({"x":np.array([5])})
y_new = model.predict(x_new)
print(f"{round(y_new[0],2)}")
#Visualization of regression model
plt.scatter(df["x"], df["y"], label="Actual Data")
plt.scatter(x_new, y_new, color ="green", label="new data")
plt.plot(df["x"], y_pred, label="Best Fit Line")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Linear Regression Best Fit")
plt.legend()
plt.show()
