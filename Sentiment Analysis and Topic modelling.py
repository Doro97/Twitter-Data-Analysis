#!/usr/bin/env python
# coding: utf-8

# In[9]:


import re
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import string

import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import STOPWORDS,WordCloud
import gensim
from gensim.models import CoherenceModel
from gensim.parsing.preprocessing import remove_stopwords
from gensim import corpora

import html
import warnings 
from textblob import TextBlob
warnings.filterwarnings("ignore", category=DeprecationWarning)

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:




df=pd.read_json('covid19.json', lines=True)
df.head()


# In[5]:


tweets_df=df[['created_at', 'id','text','source','possibly_sensitive']].copy()
tweets_df.head()


# In[33]:


def hash_removed(text):
    return re.sub("#[a-zA-Z0-9_]+" , "",text)

def mentions_removed(text):
    return re.sub("@[a-zA-Z0-9_]+" , "",text)

def url_removal(text):
    return re.sub(r'http\S+','',text)

#remove punctuation
def punct_removed(text):
    return re.sub(r'[^a-zA-Z]',' ',text)

#convert to lowercase
def lowercase(text):
    return str(text).lower()

#tokenization
def token_text(text):
    return word_tokenize(text)

#remove stopwords
stop_words=set(stopwords.words('english'))
def sw_removed(text):
    return [item for item in text if item not in stop_words]

#remove words with less than 3 characters
def less_char(text):
    return [x for x in text if len(x)>3]

#convert back to string
def convert_str(text):
    return ' '.join(text)


tweets_df['clean_text']=tweets_df['text'].apply(hash_removed).apply(mentions_removed).apply(url_removal).apply(punct_removed).apply(lowercase).apply(token_text).apply(sw_removed).apply(less_char).apply(convert_str)

tweets_df.head()


# In[35]:


#sentiment score, first value is the polarity score and second value is the subjectivity score
def sentiment(x):
    return TextBlob(x).sentiment

tweets_df['sentiment_score'] =tweets_df['clean_text'].apply(sentiment)

tweets_df.head()


# In[37]:


#polarity
def polarity(x):
    return TextBlob(x).sentiment.polarity  

tweets_df['polarity'] = tweets_df['clean_text'].apply(polarity)

#subjectivity
def subjectivity(x):
    return TextBlob(x).sentiment.subjectivity 

tweets_df['subjectivity'] = tweets_df['clean_text'].apply(subjectivity)

tweets_df.head()


# In[39]:


def text_category(p):
  if p>0.0 and p<=1.0:
    return 'positive'
  elif p>=-1.0 and p<0.0:
    return 'negative'
  
  else:
     return 'neutral'

  return p

tweets_df.info()

tweets_df['polarity_score']=tweets_df['polarity'].apply(text_category)
tweets_df.head()


# In[94]:


tweets_df['scoremap']=tweets_df['polarity_score'].map({'positive':1, 'negative':0})
tweets_df.head()


# In[40]:


#visualization using pie chart and bar chart
#pie chart
data=tweets_df.groupby(['polarity_score']).size()
data.plot(kind='pie')


# In[41]:


#bar chart
data=tweets_df.groupby(['polarity_score']).size()
data.plot(kind='bar')


# In[43]:


#dropping the neutral column

tweets_df.drop(tweets_df[tweets_df['polarity_score'] =='neutral'].index, inplace = True)

tweets_df=tweets_df.reset_index(drop=True)
tweets_df.head()

data=tweets_df.groupby(['polarity_score']).size()
data.plot(kind='bar')


# In[38]:


#visualization using word cloud
from subprocess import check_output
from wordcloud import WordCloud, STOPWORDS
plt.rcParams['figure.figsize']=(12.0,12.0)  
plt.rcParams['font.size']=12            
plt.rcParams['savefig.dpi']=100             
plt.rcParams['figure.subplot.bottom']=.1 
stopwords = set(STOPWORDS)
wordcloud = WordCloud(
                          background_color='white',
                          stopwords=stopwords,
                          max_words=500,
                          max_font_size=40, 
                          random_state=100
                         ).generate(str(tweets_df['clean_text']))
print(wordcloud)
fig = plt.figure(1)
plt.imshow(wordcloud)
plt.axis('off')
plt.show();


# In[95]:


from sklearn.model_selection import train_test_split
# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(tweets_df['clean_text'], tweets_df['scoremap'], random_state=0)


# In[96]:


from sklearn.feature_extraction.text import CountVectorizer
# Fit the CountVectorizer to the training data
vect = CountVectorizer().fit(X_train)

X_train_vectorized = vect.transform(X_train)
X_train_vectorized


# In[97]:


from sklearn.linear_model import LogisticRegression,SGDClassifier
model = LogisticRegression()
model.fit(X_train_vectorized, y_train)


# In[102]:


from sklearn.metrics import roc_curve, roc_auc_score, auc

predictions = model.predict(vect.transform(X_test))
roc_auc_Score= roc_auc_score(y_test, predictions)
roc_auc_Score

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, predictions)
roc_auc = auc(false_positive_rate, true_positive_rate)
roc_auc


# In[98]:


#using a trigram
vect = CountVectorizer(min_df=5, ngram_range=(3,3)).fit(X_train)
X_train_vectorized = vect.transform(X_train)
model = LogisticRegression()
model.fit(X_train_vectorized, y_train)

predictions = model.predict(vect.transform(X_test))
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, predictions)
roc_auc = auc(false_positive_rate, true_positive_rate)
roc_auc


# In[ ]:





# In[ ]:




