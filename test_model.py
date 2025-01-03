import pickle
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import TreebankWordTokenizer

# Define transform_text first
def transform_text(text):
    text = text.lower()
    tokenizer = TreebankWordTokenizer()
    text = tokenizer.tokenize(text)
    text = [word for word in text if word.isalnum()]
    stop_words = set(stopwords.words('english'))
    text = [word for word in text if word not in stop_words]
    ps = PorterStemmer()
    text = [ps.stem(word) for word in text]
    return " ".join(text)

# Load the model and vectorizer
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Print model information
print("Model type:", type(model))
print("Model properties:", dir(model))
if hasattr(model, 'classes_'):
    print("Classes:", model.classes_)
if hasattr(model, 'predict_proba'):
    # Test probability prediction
    msg = "WINNER!! FREE CASH PRIZE 1000$ CALL NOW!"
    transformed = transform_text(msg)
    vector = tfidf.transform([transformed])
    proba = model.predict_proba(vector)
    print("Probability distribution:", proba)

# Test with obvious spam messages
test_messages = [
    "WINNER!! FREE CASH PRIZE 1000$ CALL NOW!",
    "Congratulations! You've won a free iPhone! Click here",
    "Hi, how are you?",  # Non-spam
    "Meeting at 3pm tomorrow"  # Non-spam
]

for msg in test_messages:
    transformed = transform_text(msg)
    vector = tfidf.transform([transformed])
    prediction = model.predict(vector)[0]
    print(f"\nMessage: {msg}")
    print(f"Prediction: {'Spam' if prediction == 1 else 'Not Spam'}") 