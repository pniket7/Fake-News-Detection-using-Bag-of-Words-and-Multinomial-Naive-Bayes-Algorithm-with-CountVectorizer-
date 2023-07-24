#!/usr/bin/env python
# coding: utf-8

# In[25]:


import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import re


# In[2]:


df=pd.read_csv('/home/niket/Pictures/train.csv')


# In[3]:


df.head()


# In[4]:


# Get the Independent Features
X=df.drop('label',axis=1)


# In[5]:


X.head()


# In[6]:


# Get the Dependent features
y=df['label']


# In[7]:


y.head()


# In[8]:


df.shape


# In[10]:


df=df.dropna()


# In[11]:


df.head(10)


# In[12]:


messages=df.copy()
messages.reset_index(inplace=True)
messages.head(10)


# In[15]:


# Data Preprocessing
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
corpus = []
for i in range(0, len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['title'][i])
    review = review.lower()
    review = review.split()
    
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)


# In[16]:


# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,ngram_range=(1,3))
X = cv.fit_transform(corpus).toarray()


# In[17]:


X.shape


# In[18]:


y=messages['label']


# In[20]:


## Dividing the dataset into Train and Test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)


# In[22]:


cv.get_params()


# In[28]:


# P
import matplotlib.pyplot as plt
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[31]:


# Applying the Multinomial Naive Bayes Classification Algorithm
from sklearn.naive_bayes import MultinomialNB
classifier=MultinomialNB()
from sklearn import metrics
import numpy as np
import itertools

classifier.fit(X_train, y_train)
pred = classifier.predict(X_test)
score = metrics.accuracy_score(y_test, pred)
print("accuracy:   %0.3f" % score)
cm = metrics.confusion_matrix(y_test, pred)
plot_confusion_matrix(cm, classes=['FAKE', 'REAL'])


# In[43]:


# Step 1: Preprocess the sample sentence
sample_sentence = " The Trump campaign has repeatedly denied ties to Russia, but that didn’t stop Clinton from calling Trump a “puppet” of Russian President Vladimir Putin during the final presidential debate. The calls have grown since Friday’s FBI report to Congress about further Clinton emails being sought. "
sample_sentence = re.sub('[^a-zA-Z]', ' ', sample_sentence)
sample_sentence = sample_sentence.lower()
sample_sentence = sample_sentence.split()

sample_sentence = [ps.stem(word) for word in sample_sentence if not word in stopwords.words('english')]
sample_sentence = ' '.join(sample_sentence)

# Step 2: Convert the sample sentence to vector using CountVectorizer
sample_vector = cv.transform([sample_sentence]).toarray()

# Step 3: Predict the label of the sample sentence
predicted_label = classifier.predict(sample_vector)

# Print the prediction
if predicted_label[0] == 1:
    print("The sample sentence is classified as REAL news.")
else:
    print("The sample sentence is classified as FAKE news.")


# In[ ]:




