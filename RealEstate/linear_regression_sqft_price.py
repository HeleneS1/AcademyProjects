import numpy as np 
import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv('kc_house_data.csv')

X = np.c_[df['sqft_living']]
y = np.c_[df['price']]


import matplotlib.pyplot as plt 
plt.plot(df['sqft_living'], df['price'], 'x', alpha=0.2)


#lag tom model
lr_model = LinearRegression()
#trening av model 
lr_model.fit(X=X,y=y)
#teste modellen 
lr_model.predict([1000])        #1000 sqft 


#sørg for riktig dimensjoner/shape 
pred_x_temp = np.linspace(0,14000,100)
pred_x = pred_x_temp.reshape(-1,1)

#lag regresjons-linje
y_pred = lr_model.predict(pred_x)
plt.plot(pred_x, y_pred)

#score modellen. Svaret kommer i prosent 
print(lr_model.score(X=X, y=y))

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

y_pred = lr_model.predict(X)
mean_squared_error(y, y_pred)
mean_absolute_error(y, y_pred)
