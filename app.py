import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Streamlit UI
st.title("SMS Spam Detector")
st.write("Enter your SMS message below:")

# Text input
user_input = st.text_area("Message", "")

# Predict button
if st.button("Check if Spam"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        input_vec = vectorizer.transform([user_input])
        prediction = model.predict(input_vec)[0]
        if prediction == "spam":
            st.error("This message is **SPAM**.")
        else:
            st.success("This message is **HAM**.")
