import re
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

df = pd.read_csv('./training.1600000.processed.noemoticon.csv', encoding='latin-1', header=None)

reduced_df = df[[0,4,5]].copy()
reduced_df.columns = ['sentiment', 'user', 'tweet']
reduced_df = reduced_df.copy()
reduced_df.loc[:, 'sentiment'] = reduced_df['sentiment'].map({0: 0, 4: 1})

def clean_text(tweet):
    tweet = re.sub(r'@\w+', '', tweet)                     # eliminar menciones
    tweet = re.sub(r'http\S+|www\S+|https\S+', '', tweet) # eliminar URLs
    tweet = re.sub(r'#', '', tweet)                        # quitar #
    tweet = tweet.lower()                                  # pasar a minúsculas
    tweet = re.sub(r'[^a-záéíóúñ\s]', '', tweet)          # quitar caracteres no alfabéticos
    tweet = re.sub(r'\s+', ' ', tweet).strip()            # eliminar espacios extra
    return tweet

reduced_df['cleaned_tweet'] = reduced_df['tweet'].apply(clean_text)

stopwords = set([
    'a','about','above','after','again','against','all','am','an','and','any','are','as','at','be','because','been','before','being','below','between','both','but','by',
    'el','la','los','las','un','una','unos','unas','y','o','en','del','se','que','con','por','para','no','sí','su','como','más','pero'
])

def remove_stopwords(text):
    tokens = text.split()
    tokens = [t for t in tokens if t not in stopwords]
    return " ".join(tokens)

reduced_df['cleaned_tweet'] = reduced_df['cleaned_tweet'].apply(remove_stopwords)

X_train, X_test, y_train, y_test = train_test_split(
    reduced_df['cleaned_tweet'],
    reduced_df['sentiment'],
    test_size=0.2,
    random_state=42
)

vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

mnb = MultinomialNB()
mnb.fit(X_train_tfidf, y_train)
y_pred_mnb = mnb.predict(X_test_tfidf)
accuracy_mnb = accuracy_score(y_test, y_pred_mnb)
print(f"Multinomial Naive Bayes accuracy: {accuracy_mnb}")

lr = LogisticRegression()
lr.fit(X_train_tfidf, y_train)
y_pred_lr = lr.predict(X_test_tfidf)
accuracy_lr = accuracy_score(y_test, y_pred_lr)
print(f"Logistic Regression accuracy: {accuracy_lr}")

SVC = LinearSVC()
SVC.fit(X_train_tfidf, y_train)
y_pred_SVC = SVC.predict(X_test_tfidf)
accuracy_SVC = accuracy_score(y_test, y_pred_SVC)
print(f"SVC accuracy: {accuracy_SVC}")

max_accuracy = max(accuracy_mnb, accuracy_lr, accuracy_SVC)
best_model = None
if max_accuracy == accuracy_mnb:
    best_model = mnb
elif max_accuracy == accuracy_lr:
    best_model = lr
else:
    best_model = SVC

# Guardar el modelo
with open("best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

# Guardar también el vectorizer, que es necesario para transformar nuevos tweets
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
