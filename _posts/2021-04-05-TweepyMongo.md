## Tweepy

We'll be using a Tweepy Streaming object to snag tweets as they're posted live. We'll also filter on pretty common words such that we get a pretty general corpus of tweets.


```python
import tweepy
import pandas as pd
import csv
import re
import os
import requests
from pymongo import MongoClient
#most of these are not necessary :)
```


```python
#initialize Mongo connection
client = MongoClient()
db= client.testing
```


```python
#removed keys so twitter doesn't ban me from using their API :)
consumer_key='' #your consumer key here
consumer_secret='' #your consumer secret here
access_token='' #your access token here
access_token_secret='' #your access secret here
auth = tweepy.OAuthHandler(consumer_key,consumer_secret)
auth.set_access_token(access_token,access_token_secret)
api = tweepy.API(auth)
```


```python
#class override- apparently this is SOP for tweepy
class Listener(tweepy.StreamListener):
    
    def on_status(self,status):
        is_retweet = False
        #rt flag, start with false as default
        retweet_text= ""
        if hasattr(status, "retweeted_status"):
            is_retweet=True
            try:
                retweet_text = status.retweeted_status.extended_tweet["full_text"]
            except:
                retweet_text = status.retweeted_status.text
        
        #handles 140+ character tweets
        if hasattr(status, "extended_tweet"):
            text = status.extended_tweet["full_text"]
        else:
            text = status.text
        
        
        quoted_text =""
        if hasattr(status, "quoted_status"):
            #check if the QT was 140+ char
            if hasattr(status.quoted_status,"extended_tweet"):
                quoted_text = status.quoted_status.extended_tweet['full_text']
            else:
                quoted_text = status.quoted_status.text
                
        #some minor text-pre-processing to remove newlines and commas incase those become relevant
        remove = [',','\n']
        for i in remove:
            text = text.replace(i," ")
            quoted_text = quoted_text.replace(i," ")
            retweet_text = retweet_text.replace(i," ")
            
        
        db.processTest.insert({'handle':status.user.screen_name,'RT TF':is_retweet,'time':status.created_at,'tweet':text,'qt':quoted_text,'rt':retweet_text,'in_keyword_db':0})
    def on_error(self,status_code):
        print("Encountered streaming error (", status_code, ")")
        sys.exit()

```


```python
streamListener = Listener()
stream = tweepy.Stream(auth=api.auth, listener=streamListener, tweet_mode='extended')
tags = ["a", "the", "i", "you", "u"] #= ['python']
stream.filter(languages=["en"], track=tags)
```

Note that I don't have any way to stop this other than interrupting the process.

## Mongo Use Case: Co-occurrence Database

I'll be applying a kind of weird use case for both Mongo and SKLearn's CountVectorizer. If you are unfamiliar with a Count Vectorizer, I wouldn't worry about the details too much. If you ARE familiar with NLP- don't think about this too deeply I just wanted a fun usecase to display why I like Mongo, as this particular problem really can't be easily solved via SQL.


```python
from sklearn.feature_extraction.text import CountVectorizer
from operator import itemgetter
import numpy as np
```

Firstly, let's define some words real quick.

A **document** is our singular unit of text. It can vary in size depending on the context, here a single document refers to a single tweet.

The **corpus** is the entire body of documents.

**Stop-words** are words that we're going to filter out. Generally these are common words that do not convey a lot of meaning

One way you may think about what we'll be doing is as a intra-documental term co-occurrence database. In that, we'll be looking at the co-occurrence of words within a single document. 


```python
docs = ['this is a document',
       'this is also a document',
       'hello dog',
       'hello this is our final final document']
```


```python
count_model = CountVectorizer(ngram_range=(1,1),stop_words='english') # default unigram model
```


```python
X = count_model.fit_transform(docs)
```


```python
X
```




    <4x4 sparse matrix of type '<class 'numpy.int64'>'
    	with 7 stored elements in Compressed Sparse Row format>




```python
#unique words and their index:
count_model.vocabulary_
```




    {'document': 0, 'hello': 3, 'dog': 1, 'final': 2}




```python
#some matrix manipulation to get things in the form that is useful for our database:
Xc = (X.T * X) # this is co-occurrence matrix in sparse csr format
Xc.setdiag(0) #want to fill same word cooccurence to 0
dense = Xc.todense()
```


```python
dense
```




    matrix([[0, 0, 2, 1],
            [0, 0, 0, 1],
            [2, 0, 0, 2],
            [1, 1, 2, 0]])



This might take a bit of convincing on your own end, but if you take the vocabulary indices and count the number of times words co-occur with one another (after removing stop words and counting duplicates), you'll get the above matrix.


```python
words = ['document', 'dog', 'final', 'hello']

foo = pd.DataFrame(dense, index = words, columns=words)
```


```python
foo
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>document</th>
      <th>dog</th>
      <th>final</th>
      <th>hello</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>document</th>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>dog</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>final</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>hello</th>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



So for example, you can deduce from the above that in all our documents, the word 'final' appears in the same document as the word 'document' twice, and the word 'hello' once in relation to the word 'document'

In all, we can put together all of the above in a single function that returns our co-occurrence matrix as well as our vocabulary:


```python
def createMatrix(docs):
    count_model = CountVectorizer(ngram_range=(1,1),stop_words='english') # default unigram model
    X = count_model.fit_transform(docs)
    Xc = (X.T * X) # this is co-occurrence matrix in sparse csr format
    Xc.setdiag(0) #want to fill same word cooccurence to 0
    dense = Xc.todense()
    vocab = count_model.vocabulary_
    return dense,vocab
```

Now, how do we store such information, such that we preserve these co-occurrence relations (and can also cmmulate them with new documents), and can expand whatever storage schema as new words come?

Personally, my gut reaction when dealing with tabular data, such as our co-occurrence matrix above, would be to store it tabularly, such as through SQL. However, how would one handle the tabular storage if we need the table to expand as new words come in? No easy solution comes to my mind in the context of SQL. However, what if we didn't insist on this tabular view of things?



```python
def matrixToDict(vocab,asList):
    tempDict={}
    for i in vocab.keys():
        innerDict={}
        index1 = int(vocab.get(i))
        for j in vocab.keys():
            index2=int(vocab.get(j))
            innerDict[j]=asList[index1][index2]
        tempDict[i]=innerDict
    return tempDict

dense, vocab = createMatrix(docs)
wordDict = matrixToDict(vocab, dense.tolist())
```


```python
wordDict
```




    {'document': {'document': 0, 'hello': 1, 'dog': 0, 'final': 2},
     'hello': {'document': 1, 'hello': 0, 'dog': 1, 'final': 2},
     'dog': {'document': 0, 'hello': 1, 'dog': 0, 'final': 0},
     'final': {'document': 2, 'hello': 2, 'dog': 0, 'final': 0}}



NOW we have a dictionary, in which we have a key for each word in our vocabulary, for which each value is itself a dictionary containing the word and counts from the above co-occurrence matrix. Thus, we've managed to maintain that co-occurrence information whilst being able to move away from the tabular format. 

Perhaps more interestingly though, what we've now allowed is for us to maintain a Mongo Database in which each document corresponds to a singular word, and within each document we maintain the co-occurrences. The reason that this is helpful in getting around the issue of new words in our vocabulary is that Mongo **does not require a schema**, you can arbitrarily just throw data into your collection (and easily retrieve as long as the key:value structure is properly maintained).

To do the process of insertion into Mongo, I present the following functions to handle that interfacing:



```python
def retrieveWordsIDs(db,collectionName):
    """Called by insertUpdate, this gets the ID and words for everyword in our Mongo collection
    Input: Database object and the name of the collection
    Returns: dictionary of word:_id pairs
    """
    wordDict={}
    cursor = db[collectionName].find({},{'word':1})
    for i in cursor:
        wordDict[i['word']]=i['_id']
    return wordDict

def insertUpdate(insertList,db,collectionName):
    """Insert (if new word) or update with new document co-occurrences
    Input:
    
    insertList:     wordDict = matrixToDict(vocab,matrixList)
                    wordList = []
                    for i in wordDict:
                        wordList.append({'word':i,'counts':wordDict[i]})
    
    Above is example code to form the insertList. In essense, it's the return value of matrixToDict, but somewhat
    listified (for the sake of iteration concepts, if I recall correctly).
    
    """
    wordidDict = retrieveWordsIDs(db,collectionName)
    for i in insertList:
        if i['word'] in wordidDict.keys():
            cursor = db[collectionName].find({'word':i['word']})
            listCursor = list(cursor)
            if len(listCursor)!=1:
                return None #may as well give up because this should never happen
            currentDict = listCursor[0]
            currentCounts=currentDict['counts']
            currentKeys = list(currentCounts.keys())
            for key in list(set(currentKeys) & set(i['counts'].keys())):
                #looping over intersection of keys
                currentCounts[key]+=i['counts'][key]
            newKeys = list(set(i['counts'].keys())-set(currentKeys))
            #taking advantage of set subtraction above
            for j in newKeys:
                currentCounts[j]=i['counts'][j]
            db[collectionName].find_one_and_update({'_id':currentDict['_id']},{"$set": {'counts':currentCounts}})
        elif i['word'] not in wordidDict.keys():
            #just insert
            insertDict={}
            insertDict['word']=i['word']
            insertDict['counts']=i['counts']
            db[collectionName].insert(insertDict)
    return "Maybe it did it?"
    #TO-DO; make better return value lol
```

Putting a wrapper around everything, we can more or less summarize everything with the following function:


```python
def fullProcess(doc,db,collName):
    dense,vocab = createMatrix(doc)
    matrixList = dense.tolist()
    wordDict = matrixToDict(vocab,matrixList)
    wordList = []
    for i in wordDict:
        wordList.append({'word':i,'counts':wordDict[i]})
    insertUpdate(wordList,db,collName)
    return "good job diego!"

```

Despite the name, **full**Process (virgin descriptive name enjoyer: Diego, shouldn't your function names be descriptive of what your function actually does?!, me: No), this doesn't actully handle the full process (i.e., including pre-processing and all- more on that in a second). Just the creation of our matrix as well as insertion into Mongo.

### Text Pre-Processing

Definitely a lot of choices to be made as far as pre-processing tweets is concerned. I've provided below what I've used a couple of times on tweets to handle stemming and other pre-processing, though you will almost certainly have things that you'll want to remove or add to the function for your own purposes


```python
#below shamelessly stolen from lecture
from nltk.stem.snowball import SnowballStemmer
from pyspark.sql.functions import udf
import string

#Create the function that performs the text conversion
sn = SnowballStemmer('english')

def clean_text(text, sn=sn):
    #special case of getting rid of RT info
    rt_re = re.compile('^RT @[a-zA-Z0-9]+:')
    text = rt_re.sub(' ',text)
    #removing other @ instances
    ats_re = re.compile('^@[a-zA-Z0-9]+')
    text = ats_re.sub(' ',text)
    #remove emojis
    text = text.encode('ascii', 'ignore').decode('ascii')
    #removing punctuation
    punc_re = re.compile('[%s]' % re.escape(string.punctuation + '£'))
    text = punc_re.sub(' ', ' '+text.lower()+' ') # Pad with spaces for easier stopword removal
    # Remove numbers
    num_re = re.compile('(\\d+)')
    text = num_re.sub(' ', text)
    # Remove alphanumerical words
    alpha_num_re = re.compile("^[a-z0-9_.]+$")
    text = alpha_num_re.sub(' ', text)
    # Stemming
    text = sn.stem(text)
    # Regex for multiple spaces
    spaces_re = re.compile('\s+')
    text = spaces_re.sub(' ', text.strip())

    return text

clean_text = udf(clean_text)
```

## Querying

Lastly, let's quickly look at how to then retrieve a line from our collection, and think about what this schema might be useful for. 

I will, potentially, at a later point figure out how to upload a video onto this here website (or- could I figure out a way to host or otherwise interface my website with the database? Not sure!) as a quick proof of concept. Regardless, one could imagine that this might be a piece of functionality that helps drive, for instance, a search engine or something similar.


```python
def findTopN(word,db,collectionName,n=3):
    cursor = db[collectionName].find({'word':word})
    listCursor = list(cursor)
    if len(listCursor)!=1:
        return None #may as well give up
    counts = listCursor[0]['counts']
    returnDict = dict(sorted(counts.items(), key = itemgetter(1), reverse = True)[:n])
    return list(returnDict.keys())
```

Above, you pass the word you wish to query (the 'word' variable), and the function will retrieve the top n most co-occurring words with your queried word.


```python

```
