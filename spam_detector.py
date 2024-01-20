import streamlit as st
import joblib

st.title("SMS spam detector ⚠️")

text = st.text_area("Enter your SMS")

tfidf = joblib.load("vectorizer.pkl")
model = joblib.load("rand_forest.pkl")

text = tfidf.transform([text])

if st.button("Predict"):
    if model.predict_proba(text)[0][1] > 0.2:
        st.write("### Predicted as :red[Spam❗]")
    else:
        st.write("### Predicted as :green[Ham✅]")
