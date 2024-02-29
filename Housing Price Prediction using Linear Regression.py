#!/usr/bin/env python
# coding: utf-8

# Housing Price Prediction using Linear Regression
# 
# ![image.png](attachment:image.png)

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


import os
for dirname, _, filename in os.walk("D:\\PROJECTS\\New projects\\Housing.csv"):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # Predicting House Prices using EDA and Linear Regression Model

# In[2]:


# Importing Dataset

house_price = pd.read_csv('D:\\PROJECTS\\New projects\\Housing.csv')
house_price.head()


# In[3]:


house_price.tail()


# # Understanding the Data

# In[4]:


print('Rows and columns of the dataset : ', house_price.shape)


# In[5]:


# Getting some information


# In[6]:


house_price.info()


# In[7]:


house_price.columns


# # Checking Null values

# In[8]:


house_price.isnull().sum()


# In[9]:


# checking null vallues in visualization
import missingno as msno
msno.bar(house_price, color = 'blue', fontsize = 25);


# In[10]:


# checking duplicates
house_price.duplicated().sum()


# # Exploratory Data Analysis
1. Handling(Yes/No) Categorical Variables
# In[11]:


house_price.head()


# In[12]:


categorical_col = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']


# In[13]:


house_price[categorical_col]


# In[15]:


# Converting  Categorical (Yes/No) Variables into Numerical Variables

def binary_map(x):
    return x.map({'yes': 1, 'no': 0})


# In[17]:


house_price[categorical_col] = house_price[categorical_col].apply(binary_map)
house_price[categorical_col]


# In[18]:


house_price.head()

2. Handling Categorical Data with Dummy Variable
# In[20]:


dummy_col = pd.get_dummies(house_price['furnishingstatus'])



dummy_col.head()


# In[26]:


house_price = pd.concat([house_price, dummy_col], axis = 1)
house_price.head()


# In[27]:


house_price.drop(['furnishingstatus'], axis = 1, inplace = True)
house_price.head()


# # Splitting data into Training and Testing data

# In[28]:


house_price.columns


# In[31]:


np.random.seed(0)
hp_train, hp_test = train_test_split(house_price, train_size = 0.7, test_size = 0.3, random_state = 100)


# In[33]:


hp_train.head()


# In[34]:


hp_train.shape


# In[35]:


hp_test.shape


# # Scaling Training Data: MinMaxScaler

# In[36]:


scaler = MinMaxScaler()


# In[37]:


col_scale = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking', 'price']


# In[38]:


hp_train[col_scale] = scaler.fit_transform(hp_train[col_scale])


# # Training the Model

# In[39]:


y_train = hp_train.pop('price')
x_train = hp_train


# In[40]:


regression = LinearRegression()


# In[42]:


regression.fit(x_train, y_train)


# In[43]:


coefficients = regression.coef_
print(coefficients)


# In[44]:


score = regression.score(x_train, y_train)
print(score)


# # Scaling Test Data: MinMaxScaler

# In[45]:


col_scale = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking', 'price']


# In[46]:


hp_test[col_scale] = scaler.fit_transform(hp_test[col_scale])


# # Testing our Model

# In[47]:


y_test = hp_test.pop('price')
x_test = hp_test


# In[49]:


prediction = regression.predict(x_test)


# In[52]:


# Checking R Square value

r2 = r2_score(y_test, prediction)


# In[53]:


print(r2)


# # Comparing the actual and predicted values

# In[54]:


y_test.shape
y_test_metrics = y_test.values.reshape(-1,1)


# In[55]:


df = pd.DataFrame({'actual' : y_test_metrics.flatten(), 'predicted': prediction.flatten()})


# In[56]:


df.head(10)


# # Plotting the Graph

# In[73]:


# Creating a new figure
fig = plt.figure()

# Scatter plot of actual verses predicted values
plt.scatter(y_test, prediction, c="green", s=50, marker=".")

# Set the title and labels for the plot
plt.title('Actual vs Prediction')
plt.xlabel('Actual ', fontsize = 10)
plt.ylabel('Predicted ', fontsize = 10)


# # Mean Squared Error

# In[59]:


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, prediction)
print(" Mean Squared Error : ", mse)


# In[60]:


from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
knn_model = KNeighborsRegressor(n_neighbors = 5)
knn_model.fit(x_train, y_train)
knn_y_pred = knn_model.predict(x_test)
knn_mse = mean_squared_error(y_test, knn_y_pred)
knn_r2 = r2_score(y_test, knn_y_pred)


# In[61]:


print(" Mean Squared Error : ",knn_mse)
print("R Squared : ",knn_r2)


# In[62]:


decision_tree_model = DecisionTreeRegressor(random_state =42)
decision_tree_model.fit(x_train, y_train)
decision_tree_prediction = decision_tree_model.predict(x_test)


# In[63]:


decision_tree_mse = mean_squared_error(y_test, decision_tree_prediction)
decision_tree_r2 = r2_score(y_test, decision_tree_prediction)


# In[64]:


print("Mean Squared Error : ", decision_tree_mse)
print("R Squared : ", decision_tree_r2)


# In[66]:


algorithms = ['Linear Regression', 'KNN', 'Decision Tree']
r2_scores = [r2, knn_r2, decision_tree_r2]

plt.figure(figsize = (10,6))
plt.bar(algorithms, r2_scores, color = ['blue', 'grey', 'red'])
plt.title('R Squared Comparison')
plt.xlabel('Algorithms', fontsize = 10)
plt.ylabel('R Squared ', fontsize = 10)
plt.ylim(0,1)
plt.show()


# In[67]:


plt.figure(figsize = (10,6))
plt.scatter(y_test, knn_y_pred, c="blue", marker = ".")
plt.title(" Actual vs Predicted Median House Value")
plt.xlabel("Actual Median House Value ", fontsize = 10)
plt.ylabel(" Predicted Median House Value ", fontsize = 10)
plt.show()


# # Decision Tree Scatter Graph

# In[69]:


plt.figure(figsize = (10,6))
plt.scatter(y_test, decision_tree_prediction, c="blue", marker = ".")
plt.title(" Actual vs Predicted Median House Value ")
plt.xlabel(" Actual Median House Value ", fontsize  = 10)
plt.ylabel(" Predicted Median House Value ", fontsize = 10)
plt.show()

                       ----END----