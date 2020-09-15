import os
import pandas as pd
import numpy as np 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

os.chdir('C:/Users/Helene Stabell/Desktop/Academy/Uke 7/14_sept_2020/')
df = pd.read_csv('kc_house_data.csv')

lable = np.c_[df['price']]
features = np.c_[df[['sqft_living', 'bedrooms', 'bathrooms']]]

model1 = LinearRegression()
model1.fit(features, lable)

#new_house = np.array([[1000,3,2]])
#y_pred = model1.predict(new_house)

print(model1.score(X=features, y=lable))

y_pred = model1.predict(features)
mean_squared_error(lable, y_pred)
mean_absolute_error(lable, y_pred)


