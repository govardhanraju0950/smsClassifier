import streamlit as st
import pickle

# Load the model and vectorizer
with open('spam_classifier.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vec_file:
    vectorizer = pickle.load(vec_file)

# Streamlit app
st.title("SMS Spam Classifier")

user_input = st.text_area("Enter the message:")

if st.button("Classify"):
    if user_input:
        input_vec = vectorizer.transform([user_input])
        prediction = model.predict(input_vec)
        result = "Spam" if prediction[0] else "Ham"
        st.write(f"Prediction: {result}")
    else:
        st.write("Please enter a message.")
