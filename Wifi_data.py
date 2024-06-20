#!/usr/bin/env python
# coding: utf-8

# In[482]:


import pandas as pd
import numpy as np
import seaborn as sns
import re
from datetime import datetime
from datetime import datetime as dt
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import pointbiserialr
from scipy.stats import chi2_contingency, pointbiserialr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from operator import attrgetter

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

# clustering
from sklearn.cluster import KMeans
from time import time


# In[483]:


df = pd.read_csv("BEES Redemptions v1-1.csv")


# In[484]:


df.head()


# In[485]:


df = df.rename(columns={'abi_consumer_id': 'customer_id', 'abi_email':'email', 'abi_phone':'phone', 'abi_campaign':'campaign', 'abi_coupon_type':'coupon_type',
       'Favorite_POC_ID':'favorite_poc_id', 'POC_NAME':'poc_name', 'SALES_AREA':'sales_area', 'SALES_DISTRICT':'sales_district',
       'SALES_REGION':'sales_region', 'SALES_SUBCHANNEL':'sales_subchannel','C_YEAR':'year', 'Num_Creations':'creations',
       'Num_Redemptions':'redemptions','_col11':'event_time','abi_promotion_name':'promotion_name','abi_associated_brands':'associated_brands','abi_purchase_flag':'purchase_flag','abi_campaign_name':'campaign_name','abi_poc_id':'poc_id'})
df.head()


# In[486]:


df1 = pd.read_csv("2023.csv")


# In[487]:


df2 = pd.read_csv("2024.csv")


# In[488]:


df3 = pd.read_csv("2015-2019.csv")


# In[489]:


df4 = pd.read_csv("2020-2021.csv")


# In[490]:


df5 = pd.read_csv("2022-1.csv")


# In[491]:


df6 = pd.read_csv("2022-2.csv")


# In[492]:


df7 = pd.read_csv("2022-3.csv")


# In[493]:


wifi_data = pd.concat([df, df1, df2, df3,df4,df5,df6,df7], axis=0)
wifi_data.head()


# In[494]:


wifi_data.tail()


# In[495]:


wifi_data.shape


# In[496]:


wifi_data.columns


# #### Data Processing

# In[497]:


wifi_data.dtypes


# In[498]:


wifi_data.nunique()


# # Explorary Data Analysis

# The data comprises of wifi users and non-wifi users. For the purpose of this analysis the data will be filtered based on users who are wifi-users.

# In[499]:


wifi_data.loc[wifi_data['WiFi_User'] == 'No', 'WiFi_User'] = 'Yes'


# In[500]:


wifi_data['WiFi_User'].nunique()


# ##### Wifi Usage Analysis

# In[501]:


wifi_data['WiFi_User'].value_counts()


# In[502]:


#Age Group Analysis
plt.figure(figsize=(16, 6))
data = wifi_data['Age_Group'].value_counts()
sns.barplot(x=data.index, y=data.values, palette='rocket')
plt.title('Age Group')
plt.xlabel('Age')
plt.xticks(rotation=90)
plt.show()


# In[503]:


wifi_data['Age_Group'].value_counts()


# In[504]:


#Gender Analysis
plt.figure(figsize=(16, 6))
data = wifi_data['Gender'].value_counts()
sns.barplot(x=data.index, y=data.values, palette='rocket')
plt.title('Gender Analysis')
plt.xlabel('Gender')
plt.xticks(rotation=90)
plt.show()


# In[505]:


# Gender distribution
gender_distribution = wifi_data['Gender'].value_counts()


# In[506]:


# Plot the distribution
plt.figure(figsize=(8, 6))
gender_distribution.plot(kind='pie', autopct='%1.1f%%', colors=['lightblue', 'lightcoral', 'lightgreen'])
plt.title('Gender Distribution')
plt.ylabel('')
plt.show()


# In[507]:


wifi_data['Gender'].value_counts()


# In[508]:


#Provinces analysis
plt.figure(figsize=(16, 6))
data = wifi_data['Province'].value_counts()[:10]
sns.barplot(x=data.index, y=data.values, palette='rocket')
plt.title('Province Analysis')
plt.xlabel('Provinces')
plt.xticks(rotation=90)
plt.show()


# In[509]:


wifi_data['Province'].value_counts()[:10]


# In[510]:


#City analysis
plt.figure(figsize=(16, 6))
data = wifi_data['City'].value_counts()[:20]
sns.barplot(x=data.index, y=data.values, palette='rocket')
plt.title('Cities Analysis')
plt.xlabel('Cities')
plt.xticks(rotation=90)
plt.show()


# In[511]:


wifi_data['City'].value_counts()[:20]


# In[512]:


#Coupon Redemption analysis
plt.figure(figsize=(16, 6))
data = wifi_data['Coup_Redemption'].value_counts()
sns.barplot(x=data.index, y=data.values, palette='rocket')
plt.title('Coup_Redemption Analysis')
plt.xlabel('Coupon Redemption')
plt.xticks(rotation=90)
plt.show()


# In[513]:


wifi_data['Coup_Redemption'].value_counts()


# In[514]:


wifi_data['Coup_Creation_Only'].value_counts()


# In[515]:


wifi_data['coupon_type'].value_counts()


# In[516]:


wifi_data['campaign_name'].value_counts()


# In[517]:


wifi_data['associated_brands'].value_counts()


# In[518]:


wifi_data['campaign'].value_counts()


# In[519]:


wifi_data['promotion_name'].value_counts()


# In[520]:


wifi_data['poc_id'].value_counts()


# ### POC ID Behaviour

# In[521]:


# Poc Ids
plt.figure(figsize=(16, 6))
data = wifi_data['poc_id'].value_counts().nlargest(20)
sns.barplot(x=data.index, y=data.values, palette='rocket')
plt.title('Poc IDs')
plt.xlabel('Pocs')
plt.xticks(rotation=90)
plt.show()


# In[522]:


wifi_data['poc_id'].value_counts().nlargest(20)


# In[523]:


interactions_by_brand = wifi_data.groupby(['poc_id', 'associated_brands']).size().reset_index(name='interaction_count')

# Sort the interactions by count in descending order
interactions_by_brand_sorted = interactions_by_brand.sort_values(by=['associated_brands', 'interaction_count'], ascending=[True, False])

# Define a function to get the top n POC IDs for each associated brand
def top_n_poc_ids(group, n=10):
    return group.head(n)

# Apply the function to get the top 10 POC IDs for each associated brand
top_20_poc_ids_by_brand = interactions_by_brand_sorted.groupby('associated_brands').apply(top_n_poc_ids)

# Display the top 10 POC IDs with the highest interaction counts for each brand
print(top_10_poc_ids_by_brand)


# In[ ]:





# #### Campaign behaviour

# In[524]:


wifi_data['campaign'].value_counts()


# In[525]:


coup_redemption_distribution = wifi_data['Coup_Redemption'].value_counts()


# In[526]:


# Plot the distribution
plt.figure(figsize=(8, 6))
coup_redemption_distribution.plot(kind='pie', autopct='%1.1f%%', colors=['lightblue', 'lightcoral'])
plt.title('Coupon Redemption Distribution')
plt.ylabel('')
plt.show()


# In[527]:


wifi_data['Coup_Redemption'].value_counts()


# In[528]:


#Coupon Creation Only Analysis
coup_creation_only_distribution = wifi_data['Coup_Creation_Only'].value_counts()


# In[529]:


# Plot the distribution
plt.figure(figsize=(8, 6))
coup_creation_only_distribution.plot(kind='pie', autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'])
plt.title('Coupon Creation Only Distribution')
plt.ylabel('')
plt.show()


# In[530]:


wifi_data['Coup_Creation_Only'].value_counts()


# In[531]:


if 'poc_id' in wifi_data.columns and 'campaign' in wifi_data.columns:
    # 1. Count the number of unique POC IDs
    unique_poc_ids = wifi_data['poc_id'].nunique()
    print(f"Number of unique POC IDs: {unique_poc_ids}")

    # 2. Campaign Analysis
    campaign_distribution = wifi_data['campaign'].value_counts()

    # Plot the distribution
    plt.figure(figsize=(12, 6))
    campaign_distribution.plot(kind='bar', color='skyblue')
    plt.title('Campaign Distribution')
    plt.xlabel('Campaign')
    plt.ylabel('Number of Users')
    plt.xticks(rotation=45)
    plt.show()
else:
    print("Columns 'poc_id' and 'campaign' are not found in the dataset.")


# In[532]:


wifi_data['campaign'].value_counts()


# In[533]:


wifi_data.groupby(['campaign', 'poc_id']).size().reset_index(name='counts').head(50)


# In[534]:


if 'poc_id' in wifi_data.columns and 'campaign' in wifi_data.columns:
    # Group by campaign and count unique poc_id
    poc_per_campaign = wifi_data.groupby('campaign')['poc_id'].nunique()

    # Print the number of unique poc_ids per campaign
    print(poc_per_campaign)

    # Plot the number of unique poc_ids per campaign
    plt.figure(figsize=(12, 6))
    poc_per_campaign.plot(kind='bar', color='skyblue')
    plt.title('Number of Unique POC IDs per Campaign')
    plt.xlabel('Campaign')
    plt.ylabel('Number of Unique POC IDs')
    plt.xticks(rotation=45)
    plt.show()
else:
    print("Columns 'poc_id' and 'campaign' are not found in the dataset.")


# In[535]:


wifi_data.groupby(['campaign', 'associated_brands']).size().reset_index(name='counts').head(50)


# In[536]:


if 'poc_id' in wifi_data.columns and 'campaign' in wifi_data.columns:
    # Group by campaign and count unique poc_id
    poc_per_campaign = wifi_data.groupby('campaign')['associated_brands'].nunique()

    # Print the number of unique poc_ids per campaign
    print(poc_per_campaign)

    # Plot the number of unique poc_ids per campaign
    plt.figure(figsize=(12, 6))
    poc_per_campaign.plot(kind='bar', color='skyblue')
    plt.title('Number of Unique Associated Brands per Campaign')
    plt.xlabel('Campaign')
    plt.ylabel('Number of Unique Associated Brands')
    plt.xticks(rotation=45)
    plt.show()
else:
    print("Columns 'Associated Brands' and 'campaign' are not found in the dataset.")


# In[537]:


wifi_data.groupby(['campaign', 'coupon_type']).size().reset_index(name='counts').head(50)


# In[538]:


wifi_data.columns


# In[ ]:





# In[539]:


if 'poc_id' in wifi_data.columns and 'associated_brands' in wifi_data.columns:
    # Group by campaign and count unique poc_id
    poc_per_campaign = wifi_data.groupby('associated_brands')['poc_id'].nunique()

    # Print the number of unique poc_ids per campaign
    print(poc_per_campaign)

    # Plot the number of unique poc_ids per campaign
    plt.figure(figsize=(12, 6))
    poc_per_campaign.plot(kind='bar', color='skyblue')
    plt.title('Number of Unique Associated Brands per POC ID')
    plt.xlabel('Campaign')
    plt.ylabel('Number of Unique Associated Brands')
    plt.xticks(rotation=45)
    plt.show()
else:
    print("Columns 'Associated Brands' and 'poc_id' are not found in the dataset.")


# In[ ]:





# In[540]:


poc_per_associated_brand = wifi_data.groupby('associated_brands')['poc_id'].nunique()

#number of unique poc_ids per campaign
print(poc_per_associated_brand)

# Plot the number of unique poc_ids per campaign
plt.figure(figsize=(12, 6))
poc_per_campaign.plot(kind='bar', color='skyblue')
plt.title('Number of Unique Associated Brands per POC ID')
plt.xlabel('Campaign')
plt.ylabel('Number of Unique Associated Brands')
plt.xticks(rotation=45)
plt.show()


# In[ ]:





# ##### Coupon Behaviour

# In[541]:


Coup_Redemption_Per_Coup_Creation_Only = wifi_data.groupby('Coup_Creation_Only')['Coup_Redemption'].nunique()

#number of unique poc_ids per campaign
print(Coup_Redemption_Per_Coup_Creation_Only)

# Plot the number of unique poc_ids per campaign
plt.figure(figsize=(12, 6))
Coup_Redemption_Per_Coup_Creation_Only.plot(kind='bar', color='skyblue')
plt.title('Number of Redemptions per Coupon Creation Only')
plt.xlabel('Redemptions')
plt.xticks(rotation=45)
plt.show()


# In[542]:


wifi_data.groupby(['Coup_Creation_Only','Coup_Redemption']).size().reset_index(name='counts')


# In[ ]:





# In[543]:


associated_brands_per_coupon_type = wifi_data.groupby('coupon_type')['associated_brands'].nunique()

#number of unique poc_ids per campaign
print(associated_brands_per_coupon_type)

# Plot the number of unique poc_ids per campaign
plt.figure(figsize=(12, 6))
associated_brands_per_coupon_type.plot(kind='bar', color='skyblue')
plt.title('Number of Age Group Brands per Promotion')
plt.xlabel('Age_Group')
plt.ylabel('Number Promotions')
plt.xticks(rotation=45)
plt.show()


# In[544]:


wifi_data.groupby(['coupon_type','associated_brands']).size().reset_index(name='counts')


# In[545]:


wifi_data.groupby(['campaign', 'coupon_type']).size().reset_index(name='counts').head(50)


# In[546]:


campaign_per_coupon_type = wifi_data.groupby('')[''].nunique()

#number of unique poc_ids per campaign
print(campaign_per_coupon_type)

# Plot the number of unique poc_ids per campaign
plt.figure(figsize=(12, 6))
campaign_per_coupon_type.plot(kind='bar', color='skyblue')
plt.title('Number of Age Group Brands per Promotion')
plt.xlabel('Age_Group')
plt.ylabel('Number Promotions')
plt.xticks(rotation=45)
plt.show()


# In[ ]:


wifi_data.columns


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


Age_Group_Per_Promotion = wifi_data.groupby('campaign')['Age_Group'].nunique()

#number of unique poc_ids per campaign
print(Age_Group_Per_Promotion)

# Plot the number of unique poc_ids per campaign
plt.figure(figsize=(12, 6))
Age_Group_Per_Promotion.plot(kind='bar', color='skyblue')
plt.title('Number of Age Group Brands per Promotion')
plt.xlabel('Age_Group')
plt.ylabel('Number Promotions')
plt.xticks(rotation=45)
plt.show()


# In[ ]:


wifi_data.columns


# In[ ]:


poc_per_Coup_Redemption = wifi_data.groupby('Coup_Redemption')['poc_id'].nunique()

#number of unique poc_ids per campaign
print(poc_per_Coup_Redemption)

# Plot the number of unique poc_ids per campaign
plt.figure(figsize=(12, 6))
poc_per_Coup_Redemption.plot(kind='bar', color='skyblue')
plt.title('Number of Age Group Brands per POC ID')
plt.xlabel('Age_Group')
plt.ylabel('Number of Age Groups')
plt.xticks(rotation=45)
plt.show()


# In[ ]:


wifi_data['Province'].value_counts()


# In[ ]:


wifi_data.groupby('Province')['td_id'].nunique()


# In[ ]:


Province_poc_id = wifi_data.groupby('Province')['poc_id'].nunique()

#number of unique poc_ids per campaign
print(Province_poc_id)

# Plot the number of unique poc_ids per campaign
plt.figure(figsize=(12, 6))
Province_poc_id.plot(kind='bar', color='skyblue')
plt.title('Number of Age Group Brands per POC ID')
plt.xlabel('Age_Group')
plt.ylabel('Number of Age Groups')
plt.xticks(rotation=45)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




