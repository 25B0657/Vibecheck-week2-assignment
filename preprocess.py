import nltk
import pandas as pd
import re

#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('wordnet')
#nltk.download('punkt_tab')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

#Question 2:

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


#Question 3:
#testing the function preprocess(text)

Sentence1 = "Puppies running happily AND enjoying the beautiful gardens!"
Sentence2 = "Where are you? You'll be late for the interview."
Sentence3 = "I'm SO hyped!!!!"
Sentence4 = "ugh... everything's going wrong"
Sentence5 = "Surprisingly my password is also ROHIT@1234 !"

print("\n checking if the preprocess() is working properly or not")
print(preprocess(Sentence1))
print(preprocess(Sentence2))
print(preprocess(Sentence3))
print(preprocess(Sentence4))
print(preprocess(Sentence5))


