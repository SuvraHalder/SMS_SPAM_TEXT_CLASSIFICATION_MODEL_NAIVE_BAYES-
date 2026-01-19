import streamlit as st
import nltk
from nltk.corpus import stopwords
from string import punctuation

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
punct = set(punctuation)
stem = nltk.PorterStemmer()

def text_transform(text):
    text = text.lower()
    words = text.split()

    y = []
    for word in words:
        if word.isalnum() and word not in stop_words:
            word = stem.stem(word)
            y.append(word)

    return " ".join(y)


import pickle

with open("vectorizer.pkl", "rb") as f:
    t = pickle.load(f)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("SMS SPAM CLASSIFIER")
input_sms = st.text_area("Enter the Message")


if st.button('Predict'):

    #1 .preprocess
    transform_sms = text_transform(input_sms)

    #2 .vectoerize
    vector_input = t.transform([transform_sms])

    #predict
    result = model.predict(vector_input)[0]

    #display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")


