import streamlit as st
import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

st.title("Klasifikasi Sentimen Ulasan Netflix")

@st.cache_data
def load_data():
    return pd.read_csv("netflix_reviews.csv")

df = load_data()

def score_to_sentiment(score):
    try:
        score = int(score)
        if score >= 4:
            return "positive"
        elif score == 3:
            return "neutral"
        else:
            return "negative"
    except:
        return "invalid"

df["sentiment"] = df["score"].apply(score_to_sentiment)

def clean_text(text):
    if pd.isnull(text): return ""
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["clean_text"] = df["content"].apply(clean_text)

tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(df["clean_text"])
sequences = tokenizer.texts_to_sequences(df["clean_text"])
padded = pad_sequences(sequences, maxlen=200, padding="post")

label_map = {label: idx for idx, label in enumerate(df["sentiment"].unique())}
df["label"] = df["sentiment"].map(label_map)

X_train, X_test, y_train, y_test = train_test_split(padded, df["label"], test_size=0.2, random_state=42)

model = Sequential([
    Embedding(10000, 128),
    LSTM(64),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dense(len(label_map), activation='softmax')
])
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

st.info("Melatih model...")
model.fit(X_train, y_train, epochs=3, validation_split=0.2, verbose=0)

loss, acc = model.evaluate(X_test, y_test, verbose=0)
st.success(f"Akurasi: {acc:.4f}")

st.subheader("Distribusi Sentimen")
fig, ax = plt.subplots()
sns.countplot(x='sentiment', data=df, ax=ax)
st.pyplot(fig)

st.subheader("Prediksi Ulasan Baru")
user_input = st.text_area("Masukkan ulasan:")
if st.button("Prediksi"):
    cleaned = clean_text(user_input)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded_seq = pad_sequences(seq, maxlen=200, padding='post')
    pred = model.predict(padded_seq)
    label_pred = np.argmax(pred)
    label_name = [k for k, v in label_map.items() if v == label_pred][0]
    st.write(f"Sentimen Prediksi: **{label_name.capitalize()}**")