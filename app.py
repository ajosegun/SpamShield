import streamlit as st
import core

st.title('SpamShield')
st.subheader('''SpamShield is an Email/SMS spam classifier built using PySpark's Machine Learning library. 
                It predicts whether a given message is spam or not. The model is built using Naive Bayes 
                algorithm and preprocessed using a custom PySpark Pipeline. 
            ''')

text = st.text_area("Enter the message")
if st.button("Predict"):
    result = core.predict(text)
    
    if int(result) == 0:
        result = "Not Spam"
    else:
        result = "Spam"
    
    st.success(f'Prediction: {result}')
    
