#!/usr/bin/env python
# coding: utf-8

# # Price Prediction using Regression

# This is a tickets pricing monitoring system. It scrapes tickets pricing data periodically and stores it in a database. Ticket pricing changes based on demand and time, and there can be significant difference in price. We are creating this product mainly with ourselves in mind. Users can set up alarms using an email, choosing an origin and destination (cities), time (date and hour range picker) choosing a price reduction over mean price, etc.

# **Following is the description for columns in the dataset**<br>
# - insert_date: date and time when the price was collected and written in the database<br>
# - origin: origin city <br>
# - destination: destination city <br>
# - start_date: train departure time<br>
# - end_date: train arrival time<br>
# - train_type: train service name<br>
# - price: price<br>
# - train_class: ticket class, tourist, business, etc.<br>
# - fare: ticket fare, round trip, etc <br>

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()


# #### **Task 1: Import Dataset and create a copy of that dataset**

# In[3]:


#write code here
data = pd.read_csv('data1.csv')
df = data.copy()


# #### **Task 2: Display first five rows** 

# In[4]:


#write code here
df.head()


# #### **Task 3: Drop 'unnamed: 0' column**

# In[5]:


#write code here
df.drop('Unnamed: 0', axis=1, inplace=True)


# #### **Task 4: Check the number of rows and columns**

# In[6]:


#write code here
df.shape


# #### **Task 5: Check data types of all columns**

# In[75]:


#write code here
df.dtypes


# #### **Task 6: Check summary statistics**

# In[7]:


#write code here
df.describe()


# #### **Task 7: Check summary statistics of all columns, including object dataypes**

# In[8]:


df.describe(include="all")


# **Question: Explain the summary statistics for the above data set**

# **Answer:** 
#         From this summary statistics we conclude that there are total 215909 set of rows from which some rows of price, train_class and fare are empty or contain null values.
#         Also average price for ticket is 56 and max price for ticket is 209. There are 5 unique values for both origin and destination which means there are total 5 stations available 
#         in given dataset

# #### **Task 8: Check null values in dataset**

# In[9]:


#write code here
df.isnull().sum()


# #### **Task 9: Fill the Null values in the 'price' column.**<br>
# 

# **Data in price colomn is right skewed so we replace null values with median**

# In[10]:


sns.distplot(df['price'])


# In[11]:


#write code here
df['price'].fillna(df['price'].median(),inplace=True)


# #### **Task 10: Drop the rows containing Null values in the attributes train_class and fare**

# In[13]:


#write code here
df.dropna(subset=['train_class','fare'],inplace=True)


# #### **Task 11: Drop 'insert_date'**

# In[14]:


#write code here
df.drop('insert_date', axis=1, inplace=True)
df.shape


# **Check null values again in dataset**

# In[15]:


#write code here
df.isnull().sum()


# #### **Task 12: Plot number of people boarding from different stations**
# 

# In[99]:


#write code here
sns.countplot(x='origin', data=df);


# **Question: What insights do you get from the above plot?**

# **Answer:** 
#           Most common station for boarding is madrid, around 150,000 or more people board from this station. Ponferrada have very less number of people boarding as compared to other 
#           stations. After madrid Barcelona have more number of boardings.

# #### **Task 13: Plot number of people for the destination stations**
# 

# In[100]:


#write code here
sns.countplot(x='destination', data=df);


# **Question: What insights do you get from the above graph?**

# **Answer:**
#         Madrid is the most common station used for travelling there were also huge number of people boarding from this station and as well as huge number of people getting off this station and very less number of people are getting off from Ponferrda station 

# #### **Task 14: Plot different types of train that runs in Spain**
# 

# In[1]:


#write code here
#sns.countplot(x='train_type', data=df)
plt.figure(figsize=(15,6))
ax = sns.countplot(x='train_type', data=df)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
plt.tight_layout()
plt.show()


# **Question: Which train runs the maximum in number as compared to other train types?**

# **Answer:**
#           Ave train type in spain have maximum number as compared to other train type
# 

# #### **Task 15: Plot number of trains of different class**
# 

# In[108]:


#write code here
#plt.figure(figsize=(15,6))
ax = sns.countplot(x='train_class', data=df)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
plt.tight_layout()
plt.show()


# **Question: Which the most common train class for traveling among people in general?**

# **Answer:**  
#            Tunista is most common train class for traveling ammong people in spain.
# 

# #### **Task 16: Plot number of tickets bought from each category**
# 

# In[39]:


#write code here
sns.countplot(x='fare', data=df)


# **Question: Which the most common tickets are bought?**

# **Answer:** 
#             Promo tickets are the most common ticket bought among people

# #### **Task 17: Plot distribution of the ticket prices**

# In[32]:


#write code here
sns.distplot(df['price'])


# **Question: What readings can you get from the above plot?**

# **Answer:**   
#            From this distribution it looks like that price is positively skewed

# #### **Task 18: Show train_class vs price through boxplot**

# In[46]:


#write code here

ax = sns.boxplot(x = 'train_class', y = 'price', data = df)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
plt.tight_layout()
plt.show()


# **Question: What pricing trends can you find out by looking at the plot above?**

# **Answer:** 
#            Most of the ticket bought for class type range from 40 to 80. preferente class type have normal distribution among price and tunista is positively skewed and tunista plus is
#            negatively skewed

# #### **Task 19: Show train_type vs price through boxplot**
# 

# In[44]:


#write code here
plt.figure(figsize=(15,6))
ax = sns.boxplot(x = 'train_type', y = 'price', data = df)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="center")
plt.tight_layout()
plt.show()


# **Question: Which type of trains cost more as compared to others?**

# **Answer:** 
#            Ave train type cost more as compared to other train type.
# 
# 

# ## Feature Engineering
# 

# In[47]:


df = df.reset_index()


# **Finding the travel time between the place of origin and destination**<br>
# We need to find out the travel time for each entry which can be obtained from the 'start_date' and 'end_date' column. Also if you see, these columns are in object type therefore datetimeFormat should be defined to perform the necessary operation of getting the required time.

# **Import datetime library**

# In[48]:


#write code here
import datetime


# In[49]:


datetimeFormat = '%Y-%m-%d %H:%M:%S'
def fun(a,b):
    diff = datetime.datetime.strptime(b, datetimeFormat)- datetime.datetime.strptime(a, datetimeFormat)
    return(diff.seconds/3600.0)                  
    


# In[50]:


df['travel_time_in_hrs'] = df.apply(lambda x:fun(x['start_date'],x['end_date']),axis=1) 


# #### **Task 20: Remove redundant features**
# 

# **You need to remove features that are giving the related values as  'travel_time_in_hrs'**<br>
# *Hint: Look for date related columns*

# In[58]:


#write code here
df.drop(['start_date', 'end_date'], axis=1, inplace=True)   


# We now need to find out the pricing from 'MADRID' to other destinations. We also need to find out time which each train requires for travelling. 

# ## **Travelling from MADRID to SEVILLA**

# #### Task 21: Findout people travelling from MADRID to SEVILLA

# In[74]:


#write code here
df1 = df[(df['origin'] == 'MADRID') & (df['destination'] == 'SEVILLA')]
print("No. of people travelling from MADRID to SEVILLA = "+str(df1.shape[0]))


# #### Task 22: Make a plot for finding out travelling hours for each train type

# In[90]:


#write code here
plt.figure(figsize=(10,6))
ax = sns.barplot(x="train_type", y="travel_time_in_hrs", data=df1, ci = 0)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="center")
plt.tight_layout()
plt.show()


# #### **Task 23: Show train_type vs price through boxplot**
# 

# In[89]:


#write code here
plt.figure(figsize=(10,6))
ax = sns.boxplot(x = 'train_type', y = 'price', data = df1)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="center")
plt.tight_layout()
plt.show()


# ## **Travelling from MADRID to BARCELONA**
# 

# #### Task 24: Findout people travelling from MADRID to BARCELONA

# In[93]:


#write code here
df2= df[(df['origin'] == 'MADRID') & (df['destination'] == 'BARCELONA')]
print("No. of people travelling from MADRID to BARCELONA = "+str(df2.shape[0]))


# #### Task 25: Make a plot for finding out travelling hours for each train type

# In[94]:


#write code here
plt.figure(figsize=(10,6))
ax = sns.barplot(x="train_type", y="travel_time_in_hrs", data=df2, ci = 0)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="center")
plt.tight_layout()
plt.show()


# #### **Task 26: Show train_type vs price through boxplot**

# In[147]:


#write code here
plt.figure(figsize=(10,6))
ax = sns.boxplot(x = 'train_type', y = 'price', data = df1)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="center")
plt.tight_layout()
plt.show()


# ## **Travelling from MADRID to VALENCIA**

# #### Task 27: Findout people travelling from MADRID to VALENCIA

# In[97]:


#write code here
df3 = df[(df['origin'] == 'MADRID') & (df['destination'] == 'VALENCIA')]
print("No. of people travelling from MADRID to VALENCIA = "+str(df3.shape[0])) 


# #### Task 28: Make a plot for finding out travelling hours for each train type

# In[152]:


#write code here
plt.figure(figsize=(10,6))
ax = sns.barplot(x="train_type", y="travel_time_in_hrs", data=df3, ci = 0)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="center")
# plt.tight_layout()
plt.show()


# #### **Task 29: Show train_type vs price through boxplot**

# In[101]:


#write code here
plt.figure(figsize=(10,6))
ax = sns.boxplot(x = 'train_type', y = 'price', data = df3)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="center")
plt.tight_layout()
plt.show()


# ## **Travelling from MADRID to PONFERRADA**

# #### Task 30: Findout people travelling from MADRID to PONFERRADA

# In[103]:


#write code here
df4 = df[(df['origin'] == 'MADRID') & (df['destination'] == 'PONFERRADA')]
print("No. of people travelling from MADRID to PONFERRADA = "+str(df4.shape[0])) 


# #### Task 31: Make a plot for finding out travelling hours for each train type

# In[104]:


#write code here
plt.figure(figsize=(10,6))
ax = sns.barplot(x="train_type", y="travel_time_in_hrs", data=df4, ci = 0)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="center")
plt.tight_layout()
plt.show()


# #### **Task 32: Show train_type vs price through boxplot**

# In[106]:


#write code here
plt.figure(figsize=(10,6))
ax = sns.boxplot(x = 'train_type', y = 'price', data = df4)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="center")
plt.tight_layout()
plt.show()


# # Applying Linear  Regression

# #### Task 33: Import LabelEncoder library from sklearn 

# In[107]:


#write code here
from sklearn.preprocessing import LabelEncoder


# **Data Encoding**

# In[109]:


lab_en = LabelEncoder()
df.iloc[:,1] = lab_en.fit_transform(df.iloc[:,1])
df.iloc[:,2] = lab_en.fit_transform(df.iloc[:,2])
df.iloc[:,3] = lab_en.fit_transform(df.iloc[:,3])
df.iloc[:,5] = lab_en.fit_transform(df.iloc[:,5])
df.iloc[:,6] = lab_en.fit_transform(df.iloc[:,6])


# In[110]:


df.head()


# #### Task 34: Separate the dependant and independant variables

# In[111]:


#write code here
X = df.drop(['price'], axis=1)
Y = df[['price']]
print(X.shape)
print(Y.shape)


# #### Task 35: Import test_train_split from sklearn

# In[112]:


#write code here
from sklearn.model_selection import train_test_split


# #### Task 36:**Split the data into training and test set**

# In[113]:


#write code here
X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size=0.30, random_state=25,shuffle=True)


# #### Task 37: Import LinearRegression library from sklearn

# In[114]:


#write code here
from sklearn.linear_model import LinearRegression


# #### Task 38: Make an object of LinearRegression( ) and train it using the training data set

# In[115]:


#write code here
lr =  LinearRegression()


# In[116]:


#write code here
lr.fit(X_train, Y_train)


# #### Task 39: Find out the predictions using test data set.

# In[117]:


#write code here
lr_predict = lr.predict(X_test)


# #### Task 40: Find out the predictions using training data set.

# In[118]:


#write code here
lr_predict_train = lr.predict(X_train)


# #### Task 41: Import r2_score library form sklearn

# In[119]:


#write code here
from sklearn.metrics import r2_score


# #### Task 42: Find out the R2 Score for test data and print it.

# In[121]:


#write code here
lr_r2_test= r2_score(Y_test,lr_predict)
print("r2 score of testing "+ str(lr_r2_test))


# #### Task 43: Find out the R2 Score for training data and print it.

# In[122]:


lr_r2_train =r2_score(Y_train,lr_predict_train)
print("r2 score of training "+ str(lr_r2_train))


# Comaparing training and testing R2 scores

# In[124]:


print('R2 score for Linear Regression Training Data is: ', lr_r2_train)
print('R2 score for Linear Regression Testing Data is: ', lr_r2_test)


# # Applying Polynomial Regression

# #### Task 44: Import PolynomialFeatures from sklearn

# In[125]:


#write code here
from sklearn.preprocessing import PolynomialFeatures


# #### Task 45: Make and object of default Polynomial Features

# In[126]:


#write code here
poly_reg = PolynomialFeatures(degree=2)


# #### Task 46: Transform the features to higher degree features.

# In[127]:


#write code here
X_train_poly,X_test_poly = poly_reg.fit_transform(X_train),poly_reg.fit_transform(X_test)


# In[135]:


print(X_train.shape)
print(X_train_poly.shape)
print(X_test_poly.shape)


# #### Task 47: Fit the transformed features to Linear Regression

# In[134]:


#write code here
poly_model = LinearRegression()
poly_model.fit(X_train_poly, Y_train)


# #### Task 48: Find the predictions on the data set

# In[136]:


#write code here
y_train_predicted,y_test_predict = poly_model.predict(X_train_poly),poly_model.predict(X_test_poly)


# #### Task 49: Evaluate R2 score for training data set

# In[137]:


#evaluating the model on training dataset
#write code here
r2_train =r2_score(Y_train, y_train_predicted)


# #### Task 50: Evaluate R2 score for test data set

# In[138]:


# evaluating the model on test dataset
#write code here
r2_test = r2_score(Y_test, y_test_predict)


# Comaparing training and testing R2 scores

# In[139]:


#write code here
print ('The r2 score for training set is: ',r2_train)
print ('The r2 score for testing set is: ',r2_test)


# #### Task 51: Select the best model

# **Question: Which model gives the best result for price prediction? Find out the complexity using R2 score and give your answer.**<br>
# *Hint: Use for loop for finding the best degree and model complexity for polynomial regression model*

# In[143]:


#write code here
r2_train=[]
r2_test=[]
for i in range(1,6):
    poly_reg = PolynomialFeatures(degree=i)
    
    X_tr_poly,X_tst_poly = poly_reg.fit_transform(X_train),poly_reg.fit_transform(X_test)
    poly =LinearRegression()
    poly.fit(X_tr_poly, Y_train)
   
    y_tr_predicted,y_tst_predict = poly.predict(X_tr_poly),poly.predict(X_tst_poly)
    r2_train.append(r2_score(Y_train, y_tr_predicted))
    r2_test.append(r2_score(Y_test, y_tst_predict))
    
print ('R2 Train', r2_train)
print ('R2 Test', r2_test)


# #### Plotting the model

# In[144]:


plt.figure(figsize=(18,5))
sns.set_context('poster')
plt.subplot(1,2,1)
sns.lineplot(x=list(range(1,6)), y=r2_train, label='Training');
plt.subplot(1,2,2)
sns.lineplot(x=list(range(1,6)), y=r2_test, label='Testing');


# **Answer**

# Model with Polynomial degree 2 give the best optimum solution with accuracy of 81% and r2 score 0.81 as well as for training and testing data.
