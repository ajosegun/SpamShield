import streamlit as st
import findspark

findspark.init()
from pyspark.sql import SparkSession
import pyspark.sql.functions as F

spark = SparkSession.builder.appName('SMSSpamCollection').getOrCreate()

# load preprocessor
from pyspark.ml import PipelineModel
spam_cleaner_loaded = PipelineModel.load("spam_cleaner_pipeline")

# load model
from pyspark.ml.classification import NaiveBayesModel
predictor_loaded = NaiveBayesModel.load("spam_classifier_model")

def predict(text):
    
    '''
    Input:

    text: a string that contains the text to be predicted whether it is spam or not.
    Output:
    
    prediction: an integer value that represents the predicted class label of the input text. 
    0 indicates that the input text is not spam, while 1 indicates that it is spam.
    '''

    # preprocess the text
    # text = "URGENT! You have won a 1 week FREE membership in our £100,000 Prize Jackpot! Txt the word: CLAIM to No: 81010"
    
    # input_str = "We would like to invite you and your families to the annual company picnic that will be held on Saturday, June 15th, at the park near the office. The picnic will start at 11 am and end at 3 pm."
    input_df = spark.createDataFrame([(text,)], ["text"])
    input_df = input_df.withColumn("textLength", F.length("text"))
    
    clean_text = spam_cleaner_loaded.transform(input_df)
    
    # predict the class label
    prediction = predictor_loaded.transform(clean_text)
    prediction = int(prediction.select("prediction").first()[0])
    return prediction ## 0 = Not Spam // 1 = Spam

st.title('SpamShield')
st.subheader('''SpamShield is an Email/SMS spam classifier built using PySpark's Machine Learning library. 
                It predicts whether a given message is spam or not. The model is built using Naive Bayes 
                algorithm and preprocessed using a custom PySpark Pipeline. 
            ''')

text = st.text_area("Enter the message")
if st.button("Predict"):
    result = predict(text)
    
    if int(result) == 0:
        result = "Not Spam"
    else:
        result = "Spam"
    
    st.success(f'Prediction: {result}')
    
# predict("URGENT! You have won a 1 week FREE membership in our £100,000 Prize Jackpot! Txt the word: CLAIM to No: 81010")