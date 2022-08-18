#!/usr/bin/env python
# coding: utf-8

# In[42]:


#For manipulation
import numpy as np
import pandas as pd

#For data visualisation
import matplotlib.pyplot as plt
import seaborn as sns

#For interactivity
from ipywidgets import interact


# In[43]:


#Reading the dataset
train_data=pd.read_csv(r'C:\Users\91949\Downloads\train_data_evaluation_part_2.csv')
test_data=pd.read_csv(r'C:\Users\91949\Downloads\test_data_evaluation_part2.csv')
train_data.drop(train_data.columns[[0,1]], axis = 1 , inplace = True)
test_data.drop(test_data.columns[[0,1]], axis = 1 , inplace = True)


# In[44]:


#Shape of the dataset
print("Shape of the dataset :",train_data.shape)


# In[45]:


#Head of the dataset
train_data.head()


# In[46]:


#Checking missing values
train_data.isnull().sum()


# In[47]:


test_data.isnull().sum()


# In[48]:


# Filling Nan values
train_data = train_data.fillna(train_data.mean())


# In[49]:


train_data.isnull().sum()


# In[50]:


# Filling Nan values
test_data = test_data.fillna(test_data.mean())


# In[51]:


test_data.isnull().sum()


# In[52]:


#List of ID
train_data['Nationality'].value_counts()


# In[53]:


fig = plt.figure(figsize =(10, 7))
ax = fig.add_subplot(111)
data=[train_data['Age'],train_data['AverageLeadTime'],train_data['LodgingRevenue'],train_data['OtherRevenue']]


# Creating plot
plt.boxplot(data)
plt.xlabel('Variables')
plt.ylabel('Units')
ax.set_xticklabels(['Age','Average Lead Time','Lodging Revenue','Other Revenue'])
plt.ylim(0,5000)
 
# show plot
plt.show()


# In[54]:


ax=sns.heatmap(train_data.corr())
print(ax)


# In[55]:


plt.rcParams['figure.figsize']=(20,8)
plt.subplot(2,4,1)
sns.distplot(train_data['Age'],color='blue')
plt.xlabel('Age',fontsize=12)
plt.grid()

plt.subplot(2,4,2)
sns.distplot(train_data['AverageLeadTime'],color='lightblue')
plt.xlabel('Average Lead Time',fontsize=12)
plt.xlim(0,500)
plt.grid()

plt.subplot(2,4,3)
sns.distplot(train_data['LodgingRevenue'],color='lightgreen')
plt.xlabel('Lodging Revenue',fontsize=12)
plt.xlim(0,5000)
plt.grid()

plt.subplot(2,4,4)
sns.distplot(train_data['OtherRevenue'],color='grey')
plt.xlabel('Other Revenue',fontsize=12)
plt.xlim(0,1000)
plt.grid()

plt.subplot(2,4,5)
sns.distplot(train_data['PersonsNights'],color='red')
plt.xlabel('Persons Nights',fontsize=12)
plt.xlim(0,40)
plt.grid()

plt.subplot(2,4,6)
sns.distplot(train_data['RoomNights'],color='purple')
plt.xlabel('Room Nights',fontsize=12)
plt.xlim(0,25)
plt.grid()

plt.subplot(2,4,7)
sns.distplot(train_data['BookingsCheckedIn'],color='black')
plt.xlabel('Bookings Checked In',fontsize=12)
plt.xlim(0,10)
plt.grid()

plt.suptitle('Distribution Chart',fontsize=20)
plt.show()


# In[56]:


frames = [train_data , test_data]
data = pd.concat(frames)
data


# In[57]:


from sklearn.preprocessing import LabelEncoder
 
# Creating a instance of label Encoder.
le = LabelEncoder()
 
# Using .fit_transform function to fit label
# encoder and return encoded label
label1 = le.fit_transform(data['Nationality'])
label2 = le.fit_transform(data['DistributionChannel'])
label3 = le.fit_transform(data['MarketSegment'])

data.drop(["Nationality","DistributionChannel","MarketSegment"], axis=1, inplace=True)
 
# Appending the array to our dataFrame
# with column name 'Purchased'
data["Nationality"] = label1
data["DistributionChannel"] = label2
data["MarketSegment"] = label3
 
# printing Dataframe
data


# In[82]:


from sklearn.model_selection import train_test_split

# split into input (X) and output (Y) variables, splitting csv data
X = data.iloc[:,0:28]
Y = data.iloc[:,7]

# split X, Y into a train and test set
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.01196, random_state=0)


# In[83]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
x_train.shape


# In[97]:


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import ReLU

classifier = Sequential()


# In[98]:


classifier.add(Dense(units=27,activation='relu'))


# In[99]:


classifier.add(Dense(units=15,activation='relu'))


# In[100]:


classifier.add(Dense(1,activation='sigmoid'))


# In[101]:


classifier.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])


# In[105]:


model = classifier.fit(x_train,y_train,validation_split=0.33)


# In[110]:


y_pred = classifier.predict(x_test)


# In[109]:





# In[ ]:




