import streamlit as st
import pickle
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import TreebankWordTokenizer

def transform_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Tokenize using TreebankWordTokenizer
    tokenizer = TreebankWordTokenizer()
    text = tokenizer.tokenize(text)
    
    # Remove special characters and numbers
    text = [word for word in text if word.isalnum()]
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    text = [word for word in text if word not in stop_words]
    
    # Stemming
    ps = PorterStemmer()
    text = [ps.stem(word) for word in text]
    
    return " ".join(text)

# After loading the files
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    if not input_sms:
        st.warning('Please enter a message to classify')
    else:
        # 1. preprocess
        transformed_sms = transform_text(input_sms)
        vectorized_sms = tfidf.transform([transformed_sms])
        # 2. predict
        result = model.predict(vectorized_sms)[0]
        # 3. Display
        if result == 1:
            st.header("Spam")
        else:
            st.header("Not Spam")
