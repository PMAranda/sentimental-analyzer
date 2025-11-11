import streamlit as st
import pickle
import re

def clean_text(tweet):
    tweet = re.sub(r'@\w+', '', tweet)                     # eliminar menciones
    tweet = re.sub(r'http\S+|www\S+|https\S+', '', tweet) # eliminar URLs
    tweet = re.sub(r'#', '', tweet)                        # quitar #
    tweet = tweet.lower()                                  # pasar a minúsculas
    tweet = re.sub(r'[^a-záéíóúñ\s]', '', tweet)          # quitar caracteres no alfabéticos
    tweet = re.sub(r'\s+', ' ', tweet).strip()            # eliminar espacios extra
    return tweet

stopwords = set([
    'a','about','above','after','again','against','all','am','an','and','any','are','as','at','be','because','been','before','being','below','between','both','but','by',
    'el','la','los','las','un','una','unos','unas','y','o','en','del','se','que','con','por','para','no','sí','su','como','más','pero'
])

def remove_stopwords(text):
    tokens = text.split()
    tokens = [t for t in tokens if t not in stopwords]
    return " ".join(tokens)

def preprocess(tweet):
    cleaned = clean_text(tweet)
    cleaned = remove_stopwords(cleaned)
    return cleaned


best_model = pickle.load(open("/mount/src/sentimental-analyzer/best_model.pkl", "rb"))
vectorizer = pickle.load(open("/mount/src/sentimental-analyzer/vectorizer.pkl", "rb"))

st.title("Clasificador de Sentimientos de Tweets")

tweet_input = st.text_area("Escribe un tweet:")

if st.button("Clasificar"):
    if tweet_input.strip() == "":
        st.warning("Escribe un tweet primero.")
    else:
        # Preprocesar y vectorizar
        cleaned_tweet = preprocess(tweet_input)
        vec = vectorizer.transform([cleaned_tweet])

        # Predecir sentimiento
        prediction = best_model.predict(vec)[0]
        sentiment = "Positivo 😊" if prediction == 1 else "Negativo 😡"
        st.success(f"El sentimiento es: {sentiment}")
