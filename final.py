
# coding: utf-8

# # IBM Advance Data Science Capstone Project

# In this capstone project, My aim is to predict the box office revenue of a movie worldwide

# # Installing dependencies

# In[2]:


get_ipython().system(u'pip install wordcloud')


# # Importing the required libraries

# In[3]:



import pandas as pd # used to load data frames
import numpy as np # linear algebra
import matplotlib.pyplot as plt # for plotting beautiful graph
import seaborn as sns # used for plot interactive graph
from wordcloud import WordCloud # used to generate tagline
from collections import Counter # used to count
from sklearn import preprocessing #  used for preprocessing the data
from datetime import datetime 
import ast # abstract syntax trees
from sklearn.preprocessing import normalize # to normalize the value on same scale
from sklearn.model_selection import train_test_split #  used for splitting the dataset into training and testing 
from keras import layers
from keras import models
from sklearn.preprocessing import LabelEncoder
from keras import regularizers
from keras.metrics import mean_squared_logarithmic_error
from sklearn.model_selection import KFold


# # Loading train data

# In[4]:



import types
import pandas as pd
from botocore.client import Config
import ibm_boto3

def __iter__(self): return 0

# @hidden_cell
# The following code accesses a file in your IBM Cloud Object Storage. It includes your credentials.
# You might want to remove those credentials before you share your notebook.
client_04a50076237749509ea263c67d7be9af = ibm_boto3.client(service_name='s3',
    ibm_api_key_id='_bGD9A17kJTEU8FMnqGABi__kQSsm4Zv8pxZ_3FbK38K',
    ibm_auth_endpoint="https://iam.bluemix.net/oidc/token",
    config=Config(signature_version='oauth'),
    endpoint_url='https://s3-api.us-geo.objectstorage.service.networklayer.com')

body = client_04a50076237749509ea263c67d7be9af.get_object(Bucket='capstone-donotdelete-pr-7osfjvfakpvd11',Key='train.csv')['Body']
# add missing __iter__ method, so pandas accepts body as file-like object
if not hasattr(body, "__iter__"): body.__iter__ = types.MethodType( __iter__, body )

df_train = pd.read_csv(body)
df_train.head()



# # Loading test data

# In[5]:



body = client_04a50076237749509ea263c67d7be9af.get_object(Bucket='capstone-donotdelete-pr-7osfjvfakpvd11',Key='test.csv')['Body']
# add missing __iter__ method, so pandas accepts body as file-like object
if not hasattr(body, "__iter__"): body.__iter__ = types.MethodType( __iter__, body )

df_test = pd.read_csv(body)
df_test.head()



# # Getting a basic ideas about the data
# 

# In[6]:


# shape of training data
df_train.shape


# In[7]:


# shape of test data
df_test.shape


# In[8]:


#information about training data
df_train.info()


# So far, we are unfimilar with the data and its features, and what they are representing. In real world, we come acroos to many different problems where we don't know the meanining of features but to imagine in our minds. What we must know is the distribution of data like variance, standart deviation, number of sample (count) or max min values. These type of information helps to understand the data, how normally distributed it is, or it has skewed distribution.

# In[9]:


df_train.describe(include='all')


# In[10]:


#count the missing values
df_train.isna().sum().sort_values(ascending=False)


# In[72]:


# Visulaisation of missing data
missing=df_train.isna().sum().sort_values(ascending=False)
sns.barplot(missing[:8],missing[:8].index)
plt.show()


# In[12]:


plt.style.use('seaborn')


# In[13]:


import ast
dict_columns = ['belongs_to_collection', 'genres', 'production_companies',
                'production_countries', 'spoken_languages', 'Keywords', 'cast', 'crew']

def text_to_dict(df):
    for column in dict_columns:
        df[column] = df[column].apply(lambda x: {} if pd.isna(x) else ast.literal_eval(x) )
    return df
        
dfx = text_to_dict(df_train)
for col in dict_columns:
       df_train[col]=dfx[col]


# In[14]:


df_train['belongs_to_collection'].apply(lambda x:len(x) if x!= {} else 0).value_counts()


# Only 604 films belong to some collections

# # Visualisation

# In[15]:


collections=df_train['belongs_to_collection'].apply(lambda x : x[0]['name'] if x!= {} else '?').value_counts()[1:15]
sns.barplot(collections,collections.index)
plt.show()


# We can observe that james bond collection films,friday the 13th
# 

# In[16]:


df_train['tagline'].apply(lambda x:1 if x is not np.nan else 0).value_counts()


# In[17]:


plt.figure(figsize=(10,10))
taglines=' '.join(df_train['tagline'].apply(lambda x:x if x is not np.nan else ''))

wordcloud = WordCloud(max_font_size=None, background_color='white', collocations=False,
                      width=1200, height=1000).generate(taglines)
plt.imshow(wordcloud)
plt.title('Taglines')
plt.axis("off")
plt.show()


# In[18]:


keywords=df_train['Keywords'].apply(lambda x: ' '.join(i['name'] for i in x) if x != {} else '')
plt.figure(figsize=(10,10))
data=' '.join(words for words in keywords)
wordcloud = WordCloud(max_font_size=None, background_color='white', collocations=False,
                      width=1200, height=1000).generate(data)
plt.imshow(wordcloud)
plt.title('Taglines')
plt.axis("off")
plt.show()


# Production companies
#  
#  The most famous production companies and the number films released by each

# In[19]:


# 20 most common production companies
x=df_train['production_companies'].apply(lambda x : [x[i]['name'] for i in range(len(x))] if x != {} else []).values
Counter([i for j in x for i in j]).most_common(20)


# In[20]:


# 20 most common production countries
countries=df_train['production_countries'].apply(lambda x: [i['name'] for i in x] if x!={} else []).values
count=Counter([j for i in countries for j in i]).most_common(10)
sns.barplot([val[1] for val in count],[val[0] for val in count])


# In[21]:


# count of different different languages
df_train['spoken_languages'].apply(lambda x:len(x) if x !={} else 0).value_counts()


# This indicates the number of languages spoken in a film
# 

# In[22]:


# 5 most common spoken languages
lang=df_train['spoken_languages'].apply(lambda x: [i['name'] for i in x] if x != {} else [])
count=Counter([i for j in lang for i in j]).most_common(5)
sns.barplot([val[1] for val in count],[val[0] for val in count])


# In[23]:


# 10 most common genres
genre=df_train['genres'].apply(lambda x: [i['name'] for i in x] if x != {} else [])
count=Counter([i for j in genre for i in j]).most_common(10)
sns.barplot([val[1] for val in count],[val[0] for val in count])


# In[24]:


dfx = text_to_dict(df_test)
for col in dict_columns:
       df_test[col]=dfx[col]


# Revenue
# 
# This is the target variable
# 
# We will inspect the distribution first.

# In[25]:


plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.title('skewed data')
sns.distplot(df_train['revenue'])
plt.subplot(1,2,2)
plt.title('log transformation')
sns.distplot(np.log(df_train['revenue']))
plt.show()


# In[26]:


df_train['log_revenue']=np.log1p(df_train['revenue'])


# In[27]:


plt.subplots(figsize=(10,5))
plt.subplot(1,2,1)
plt.hist(df_train['revenue'],bins=10,color='g')
plt.title('skewed data')
plt.subplot(1,2,2)
plt.hist(np.log(df_train['revenue']),bins=10,color='g')
plt.title('log transformation')
plt.show()


# In[28]:


df_train['revenue'].describe()


# Budget

# In[29]:


plt.subplots(figsize=(10,5))
plt.subplot(1,2,1)
plt.hist(df_train['budget']+1,bins=10,color='g')
plt.title('skewed data')
plt.subplot(1,2,2)
plt.hist(np.log(df_train['budget']+1),bins=10,color='g')
plt.title('log transformation')
plt.show()


# Revenue vs Budget

# In[30]:


plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
sns.scatterplot(df_train['budget'],df_train['revenue'])
plt.subplot(1,2,2)
sns.scatterplot(np.log1p(df_train['budget']),np.log1p(df_train['revenue']))
plt.show()


# In[31]:


df_train['log_budget']=np.log1p(df_train['budget'])
df_test['log_budget']=np.log1p(df_train['budget'])


# # Popularity

# In[32]:


plt.hist(df_train['popularity'],bins=30,color='violet')
plt.show()


# Popularity vs Revenue

# In[33]:


sns.scatterplot(df_train['popularity'],df_train['revenue'],color='violet')
plt.show()


# # Extracting date features¶
# 

# In[34]:


def date(x):
    x=str(x)
    year=x.split('/')[2]
    if int(year)<19:
        return x[:-2]+'20'+year
    else:
        return x[:-2]+'19'+year
df_train['release_date']=df_train['release_date'].fillna('1/1/90').apply(lambda x: date(x))
df_test['release_date']=df_test['release_date'].fillna('1/1/90').apply(lambda x: date(x))


# In[35]:


#from datetime import datetime
df_train['release_date']=df_train['release_date'].apply(lambda x: datetime.strptime(x,'%m/%d/%Y'))
df_test['release_date']=df_test['release_date'].apply(lambda x: datetime.strptime(x,'%m/%d/%Y'))


# In[36]:


df_train['release_day']=df_train['release_date'].apply(lambda x:x.weekday())
df_train['release_month']=df_train['release_date'].apply(lambda x:x.month)
df_train['release_year']=df_train['release_date'].apply(lambda x:x.year)


# In[37]:


df_test['release_day']=df_test['release_date'].apply(lambda x:x.weekday())
df_test['release_month']=df_test['release_date'].apply(lambda x:x.month)
df_test['release_year']=df_test['release_date'].apply(lambda x:x.year)


# # no of movies release in per Year

# In[38]:


plt.figure(figsize=(18,12))
sns.countplot(df_train['release_year'].sort_values())
plt.title("no of movies release in a Year",fontsize=20)
loc, labels = plt.xticks()
plt.xticks(fontsize=12,rotation=90)
plt.show()


# # no of movies release in each month and in each day

# In[39]:


plt.figure(figsize=(15,5))
sns.countplot(df_train['release_month'].sort_values())
plt.title("no of movies release in each month",fontsize=20)
loc, labels = plt.xticks()
loc, labels = loc, ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
plt.xticks(loc, labels,fontsize=20)
plt.show()


plt.figure(figsize=(15,5))
sns.countplot(df_train['release_day'].sort_values())
plt.title("no of movies release every day",fontsize=20)
plt.xticks(fontsize=15)
plt.show()


# Revenue vs Release year

# In[40]:


plt.figure(figsize=(15,8))
yearly=df_train.groupby(df_train['release_year'])['revenue'].agg('mean')
plt.plot(yearly.index,yearly)
plt.xlabel('year')
plt.ylabel("Revenue")
plt.savefig('fig')


# # Runtime

# In[41]:


plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
sns.distplot(np.log1p(df_train['runtime'].fillna(0)))

plt.subplot(1,2,2)
sns.scatterplot(np.log1p(df_train['runtime'].fillna(0)),np.log1p(df_train['revenue']))


# In[42]:


df_train['homepage'].value_counts().sort_values(ascending=False)[:5]


# In[43]:


genres=df_train.loc[df_train['genres'].str.len()==1][['genres','revenue','budget','popularity','runtime']].reset_index(drop=True)
genres['genres']=genres.genres.apply(lambda x :x[0]['name'])


# In[44]:


genres=genres.groupby(genres.genres).agg('mean')


# In[45]:


plt.figure(figsize=(15,10))
plt.subplot(2,2,1)
sns.barplot(genres['revenue'],genres.index)

plt.subplot(2,2,2)
sns.barplot(genres['budget'],genres.index)

plt.subplot(2,2,3)
sns.barplot(genres['popularity'],genres.index)

plt.subplot(2,2,4)
sns.barplot(genres['runtime'],genres.index)


# # Crew

# In[46]:


crew=df_train['crew'].apply(lambda x:[i['name'] for i in x] if x != {} else [])
Counter([i for j in crew for i in j]).most_common(15)


# # Cast

# In[47]:


cast=df_train['cast'].apply(lambda x:[i['name'] for i in x] if x != {} else [])
Counter([i for j in cast for i in j]).most_common(15)


# # Feature Engineering

# In[48]:


def  prepare_data(df):
    df['_budget_runtime_ratio'] = (df['budget']/df['runtime']).replace([np.inf,-np.inf,np.nan],0)
    df['_budget_popularity_ratio'] = df['budget']/df['popularity']
    df['_budget_year_ratio'] = df['budget'].fillna(0)/(df['release_year']*df['release_year'])
    df['_releaseYear_popularity_ratio'] = df['release_year']/df['popularity']
    df['_releaseYear_popularity_ratio2'] = df['popularity']/df['release_year']
    df['budget']=np.log1p(df['budget'])
    
    df['collection_name']=df['belongs_to_collection'].apply(lambda x: x[0]['name'] if x != {} else 0)
    df['has_homepage']=0
    df.loc[(pd.isnull(df['homepage'])),'has_homepage']=1
    
    le=LabelEncoder()
    le.fit(list(df['collection_name'].fillna('')))
    df['collection_name']=le.transform(df['collection_name'].fillna('').astype(str))
    
    le=LabelEncoder()
    le.fit(list(df['original_language'].fillna('')))
    df['original_language']=le.transform(df['original_language'].fillna('').astype(str))
    
    df['_num_Keywords'] = df['Keywords'].apply(lambda x: len(x) if x != {} else 0)
    df['_num_cast'] = df['cast'].apply(lambda x: len(x) if x != {} else 0)
    
    df['isbelongto_coll']=0
    df.loc[pd.isna(df['belongs_to_collection']),'isbelongto_coll']=1
    
    df['isTaglineNA'] = 0
    df.loc[df['tagline'] == 0 ,"isTaglineNA"] = 1 

    df['isOriginalLanguageEng'] = 0 
    df.loc[ df['original_language'].astype(str) == "en" ,"isOriginalLanguageEng"] = 1
    
    df['ismovie_released']=1
    df.loc[(df['status']!='Released'),'ismovie_released']=0
    
    df['no_spoken_languages']=df['spoken_languages'].apply(lambda x: len(x))
    df['original_title_letter_count'] = df['original_title'].str.len() 
    df['original_title_word_count'] = df['original_title'].str.split().str.len() 


    df['title_word_count'] = df['title'].str.split().str.len()
    df['overview_word_count'] = df['overview'].str.split().str.len()
    df['tagline_word_count'] = df['tagline'].str.split().str.len()
    
    
    df['collection_id'] = df['belongs_to_collection'].apply(lambda x : np.nan if len(x)==0 else x[0]['id'])
    df['production_countries_count'] = df['production_countries'].apply(lambda x : len(x))
    df['production_companies_count'] = df['production_companies'].apply(lambda x : len(x))
    df['cast_count'] = df['cast'].apply(lambda x : len(x))
    df['crew_count'] = df['crew'].apply(lambda x : len(x))

    df['genders_0_crew'] = df['crew'].apply(lambda x: sum([1 for i in x if i['gender'] == 0]))
    df['genders_1_crew'] = df['crew'].apply(lambda x: sum([1 for i in x if i['gender'] == 1]))
    df['genders_2_crew'] = df['crew'].apply(lambda x: sum([1 for i in x if i['gender'] == 2]))

    for col in  ['genres', 'production_countries', 'spoken_languages', 'production_companies'] :
        df[col] = df[col].map(lambda x: sorted(list(set([n if n in train_dict[col] else col+'_etc' for n in [d['name'] for d in x]])))).map(lambda x: ','.join(map(str, x)))
        temp = df[col].str.get_dummies(sep=',')
        df = pd.concat([df, temp], axis=1, sort=False)
    df.drop(['genres_etc'], axis = 1, inplace = True)
    
    cols_to_normalize=['runtime','popularity','budget','_budget_runtime_ratio','_budget_year_ratio','_budget_popularity_ratio','_releaseYear_popularity_ratio',
    '_releaseYear_popularity_ratio2','_num_Keywords','_num_cast','no_spoken_languages','original_title_letter_count','original_title_word_count',
    'title_word_count','overview_word_count','tagline_word_count','production_countries_count','production_companies_count','cast_count','crew_count',
    'genders_0_crew','genders_1_crew','genders_2_crew']
    for col in cols_to_normalize:
        print(col)
        x_array=[]
        x_array=np.array(df[col].fillna(0))
        X_norm=normalize([x_array])[0]
        df[col]=X_norm
    
    df = df.drop(['belongs_to_collection','genres','homepage','imdb_id','overview','id'
    ,'poster_path','production_companies','production_countries','release_date','spoken_languages'
    ,'status','title','Keywords','cast','crew','original_language','original_title','tagline', 'collection_id'
    ],axis=1)
    
    df.fillna(value=0.0, inplace = True) 

    return df


# In[49]:


def get_json(df):
    global dict_columns
    result=dict()
    for col in dict_columns:
        d=dict()
        rows=df[col].values
        for row in rows:
            if row is None: continue
            for i in row:
                if i['name'] not in d:
                    d[i['name']]=0
                else:
                    d[i['name']]+=1
            result[col]=d
    return result
    
    

    
train_dict=get_json(df_train)
test_dict=get_json(df_test)


# In[50]:


df_train.shape


# In[51]:


for col in dict_columns :
    
    remove = []
    train_id = set(list(train_dict[col].keys()))
    test_id = set(list(test_dict[col].keys()))   
    
    remove += list(train_id - test_id) + list(test_id - train_id)
    for i in train_id.union(test_id) - set(remove) :
        if train_dict[col][i] < 10 or i == '' :
            remove += [i]
    for i in remove :
        if i in train_dict[col] :
            del train_dict[col][i]
        if i in test_dict[col] :
            del test_dict[col][i]


# In[52]:


df_test['revenue']=np.nan
all_data=prepare_data((pd.concat([df_train,df_test]))).reset_index(drop=True)
train=all_data.loc[:df_train.shape[0]-1,:]
test=all_data.loc[df_train.shape[0]:,:]
print(train.shape)


# In[53]:


all_data.head()


# In[54]:


train.drop('revenue',axis=1,inplace=True)


# In[55]:


y=train['log_revenue']
X=train.drop(['log_revenue'],axis=1)


# # Splitting training and test data

# In[56]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1)
kfold=KFold(n_splits=3,random_state=42,shuffle=True)


# In[57]:


X.columns


# # keras Model 

# In[58]:


from keras import optimizers

    
model=models.Sequential()
model.add(layers.Dense(356,activation='relu',kernel_regularizer=regularizers.l1(.001),input_shape=(X.shape[1],)))
#model.add(layers.Dropout(0.1))
model.add(layers.Dense(256,kernel_regularizer=regularizers.l1(.001),activation='relu'))
model.add(layers.Dense(1))
model.compile(optimizer=optimizers.rmsprop(lr=.001),loss='mse',metrics=['mean_squared_logarithmic_error'])


# In[59]:


epochs=40


# In[60]:


hist=model.fit(X_train,y_train,epochs=epochs,validation_data=(X_test,y_test))


# no of epochs vs mean absolute error

# In[61]:


plt.figure(figsize=(15,5))
mae=hist.history['mean_squared_logarithmic_error']
plt.plot(range(1,epochs),mae[1:],label='mae')
plt.xlabel('epochs')
plt.ylabel('mean_abs_error')
mae=hist.history['val_mean_squared_logarithmic_error']
plt.plot(range(1,epochs),mae[1:],label='val_mae')
plt.legend()


# In[62]:


mae=hist.history['loss']
plt.plot(range(1,epochs),mae[1:],label='trraining loss')
plt.xlabel('epochs')
plt.ylabel('loss')
mae=hist.history['val_loss']
plt.plot(range(1,epochs),mae[1:],label='val_loss')
plt.legend()


# Average mae

# In[63]:


print(mae)
sum(mae)/len(mae)


# # Random Forest Regressor

# In[66]:


import math
def rmse(x,y): return math.sqrt(((x-y)**2).mean())

# print_score function depending on the evaluation metric: rmse
def print_score(m):
    res = [rmse(m.predict(X_train), y_train), rmse(m.predict(X_test), y_test),
                m.score(X_train, y_train), m.score(X_test, y_test)]
    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)
    print(res)


# In[67]:


from sklearn.ensemble import RandomForestRegressor



clf_rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=10, max_features=0.5, n_jobs=-1, oob_score=True)      
clf_rf.fit(X_train,y_train)

print_score(clf_rf)


# # Summary
# 

# In this notebook we implment two models first one is Keras and Second one in non deep learning models that is Random forest 
# regressor. We get better results on deep learninng models 
