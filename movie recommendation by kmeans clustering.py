#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder,MinMaxScaler,StandardScaler
import random
from factor_analyzer import FactorAnalyzer


# In[2]:


movie = pd.read_csv(r"C:\Users\91983\Desktop\PGA\ml\movie review\movies data.csv")
ratings = pd.read_csv(r"C:\Users\91983\Desktop\PGA\ml\movie review\ratings data.csv")


# In[3]:


print("ratings")
print(ratings.head())
print("------------------------------------------------------------------------------------")
print(ratings.info())
print("------------------------------------------------------------------------------------")
print("Movie")
print(movie.head())
print("------------------------------------------------------------------------------------")
print(movie.info())


# In[4]:


#dropping unnecessary columns
movie.drop("Unnamed: 0",axis=1,inplace=True)
ratings.drop(["Unnamed: 0","Timestamp"],axis=1,inplace=True)


# In[5]:


print(movie.columns)
print(ratings.columns)


# In[6]:


len(ratings)


# In[7]:


len(movie)


# In[8]:


#merging df
df = pd.merge(movie,ratings,how="outer",on="MovieID")


# In[9]:


df = df.sample(10000)


# In[10]:


len(df)


# In[11]:


df.head()


# In[12]:


df.info()


# In[13]:


df.isnull().sum()


# In[14]:


len(df.Genres.value_counts())


# In[15]:


df.Genres.value_counts()


# In[16]:


# Data Processing
# Converting Genres into different columns 
# Here we just create columns and put there initial value as 0
x = df.Genres
a = list()
for i in x:
    abc = i
    a.append(abc.split('|'))
a = pd.DataFrame(a)   
b = a[0].unique()
for i in b:
    df[i] = 0
df.head(2)


# In[17]:


# we assign 1 to all the columns which are present in the Genres
for i in b:
    df.loc[df['Genres'].str.contains(i), i] = 1


# In[18]:


df.head()


# In[19]:


df.drop(["Genres","Title"],axis =1,inplace=True)


# In[20]:


df.head()


# In[21]:


df.isnull().sum()


# In[22]:


df.dropna(inplace=True)


# In[23]:


df.isnull().sum()


# In[24]:


df.head()


# In[25]:


kmeans = KMeans(20)
kmeans.fit(df)


# In[26]:


kmeans.inertia_


# In[27]:


from sklearn.metrics import silhouette_score


# In[28]:


inertia=[]
sil_score =[]
for i in range(2,10):
    kmeans = KMeans(i)
    kmeans.fit(df)
    labels = kmeans.predict(df)
    iner_iter = kmeans.inertia_
    inertia.append(iner_iter)
    sil_score.append(silhouette_score(df,labels))


# In[29]:


number_clusters = range(2,10)
plt.plot(number_clusters,inertia)
plt.scatter(number_clusters,inertia)


# In[30]:


number_clusters = range(2,10)
plt.plot(number_clusters,sil_score)
plt.scatter(number_clusters,sil_score)


# In[31]:


# so by elbow method by analyzing inertia and crosschecking sil score it is safe to say that our the optimum number of 
# clusters for this model is 5


# In[32]:


kmeans = KMeans(5)
kmeans.fit(df)


# In[33]:


kmeans.labels_


# In[34]:


kmeans.fit_predict(df)


# In[35]:


df['Cluster'] = kmeans.labels_


# In[36]:


df.head()

when we merged the dataframe a single move is alloted to diffrent clusters to overcome this problem we will allot
all the similar movies to a single cluster this occurs most of the times
# In[37]:


df.Cluster.value_counts()

list1 = []
def abc(group):
    x = pd.DataFrame(group)
    xx = pd.DataFrame(x["Cluster"].value_counts())
    xxx = x.index
    i = [x["MovieID"][xxx[0]],int(xx.idxmax())] #idx max returns the max values for each columns
    i.append(list1)df.groupby("MovieID").apply(lambda a: abc(a))
# In[38]:


l1 = []
def arc(ra):
    a = pd.DataFrame(ra)
    b = pd.DataFrame(a['Cluster'].value_counts())
    d = a.index 
    c = [a['MovieID'][d[0]],int(b.idxmax())]
    l1.append(c)
df.groupby("MovieID").apply(lambda x: arc(x))

l1 = pd.DataFrame(l1)
l1.head()


# In[39]:


l1.rename(columns = {0:'MovieID',1:'Cluster'},inplace=True)


# In[40]:


l1.Cluster.value_counts()


# In[41]:


l1


# In[42]:


l1.duplicated().sum()


# In[43]:


l1.drop_duplicates(inplace=True)

now we add the orignal data set with our df l1 with movie id
# In[44]:


movie1 = pd.read_csv(r"C:\Users\91983\Desktop\PGA\ml\movie review\movies data.csv")


# In[45]:


new_data = pd.merge(l1 , movie1 , how='inner', on='MovieID')


# In[46]:


new_data.isnull().sum()


# In[47]:


new_data


# In[48]:


new_data.Cluster.value_counts()


# In[49]:


new_data.drop("Unnamed: 0",axis=1,inplace=True)


# In[50]:


new_data.info()


# In[51]:


#This function select the cluster for a user according the the user choice
def select_c():
    global l
    print('Select The Movies Id you would like to watch:')
    l=[]
    for i in range(10):
        l.append(random.randint(0,1819))
    for i in l:
        print(new_data['MovieID'][i] , new_data['Title'][i], new_data['Genres'][i],sep='--->')
    print('--------------------------------------------------------------------')
    l = int(input())
    l = new_data['Cluster'][new_data.MovieID == l]


# In[52]:


# This is the main function which recommend you movies
def main():
    ans = False
    while not ans:
        select_c()
        print(new_data['Title'][new_data.Cluster == int(l)].sample(n=10))
        print('--------------------------------------------------------------------')
        print('Do you like these movies(y/n)')
        abc = input()
        while ((abc =='y') or (abc == 'Y')):          
            print(new_data['Title'][new_data.Cluster == int(l)].sample(n=10))
            print('--------------------------------------------------------------------')
            print('Want more!!!!(y/n)')
            abc = input()
            if ((abc =='N') or (abc == 'n')):
                ans =True


# In[ ]:


main()

