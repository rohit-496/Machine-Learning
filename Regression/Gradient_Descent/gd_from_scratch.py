from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


X,y = make_regression(n_samples=100, n_features=1, n_informative=1, n_targets=1,noise=20,random_state=13)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)
#In this we separate the model training data and the model test data in the 80:20 ratio which gives the prediction and r2 score on the basis of that we are able to conclude is out model accurate or not
'''training on the basis od linear regression'''
# lr = LinearRegression()

# lr.fit(X_train,y_train) #trained on the basis of trning data
# print(lr.coef_)
# print(lr.intercept_)

# y_pred = lr.predict(X_test) #prediction for the left 20% data to test the model
# r2_score(y_test,y_pred)

class GDRegressor:
    
    def __init__(self,learning_rate,epochs):
        self.m = 100 #initialization random value
        self.b = -120 #initialization random value
        self.lr = learning_rate
        self.epochs = epochs
        
    def fit(self,X,y):
        # calcualte the b using GD
        for i in range(self.epochs):
            loss_slope_b = -2 * np.sum(y - self.m*X.ravel() - self.b)  #ravel converts 2D data  into 1D array for proper calculation
            loss_slope_m = -2 * np.sum((y - self.m*X.ravel() - self.b)*X.ravel()) #loss function derivative
            
            self.b = self.b - (self.lr * loss_slope_b)
            self.m = self.m - (self.lr * loss_slope_m)
        print(self.m,self.b)
        
    def predict(self,X):
        return self.m * X + self.b
    
    gd = GDRegressor(0.001,50) #initialized constructor

    gd.fit(X_train,y_train)
    y_pred = gd.predict(X_test)

    r2_score(y_test,y_pred)