import nltk
import pandas as pd
import re

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


df = pd.read_csv("test.txt" , sep = ';', names = ['text', 'emotion'])

#Question 4:

#The first 10 rows
print("\nThe first 10 rows or the given dataset are following")
print(df.head(10))

#Count of each emotion
print("\nHere is the list of frequency of each emotion")
print(df['emotion'].value_counts())

#The average word count per sentence before preprocessing
df['word_count'] = df['text'].apply(
    lambda x: len(str(x).split())
)

print("\nAverage word count per sentence")
print(df['word_count'].mean()) 

#Question 5:

#Preprocess fuction from question2

def preprocess(text):

    # step 1 is convert the text in lowercase
    text = text.lower()

    # step 2 is remove punctuation
    text = re.sub(r'[^\w\s]', '', text)

    # step 3 is perform tokenization on the text
    tokens = word_tokenize(text)

    # step 4 is stop word removal from the english text
    stop_words = set(stopwords.words('english'))

    filtered_words = []

    for word in tokens:
        if word not in stop_words:
            filtered_words.append(word)

    # step 5 is perform lemmatization on the text
    lemmatizer = WordNetLemmatizer()

    lemmatized_words = []

    for word in filtered_words:
        lemmatized_words.append(lemmatizer.lemmatize(word))

    return lemmatized_words


#Selecting 3 different sentense of different mood

joy = df[df['emotion'] == 'joy'].iloc[0]['text']
sad = df[df['emotion'] == 'sadness'].iloc[0]['text']
anger= df[df['emotion'] == 'anger'].iloc[0]['text']

print("JOY:", joy)
print("\nAfter applying preprocess", preprocess(joy))

print("\nSADNESS:", sad)
print("\nAfter applying preprocess", preprocess(sad))

print("\nANGER:", anger)
print("\nAfter applying preprocess", preprocess(anger))