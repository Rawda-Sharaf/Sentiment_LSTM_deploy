import streamlit as st
import numpy as np
import tensorflow as tf
import nltk
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import string
import unicodedata
import html

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))


def remove_special_chars(text):
    re1 = re.compile(r'  +')
    x1 = text.lower().replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace(
        'nbsp;', ' ').replace('#36;', '$').replace('\\n', "\n").replace('quot;', "'").replace(
        '<br />', "\n").replace('\\"', '"').replace('<unk>', 'u_n').replace(' @.@ ', '.').replace(
        ' @-@ ', '-').replace('\\', ' \\ ')
    return re1.sub(' ', html.unescape(x1))

def remove_non_ascii(text):
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')

def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

def to_lowercase(text):
    return text.lower()

def replace_numbers(text):
    return re.sub(r'\d+', '', text)

def remove_stopwords(words):
    return [word for word in words if word not in stop_words]

def normalize_text(text):
    text = remove_special_chars(text)
    text = remove_non_ascii(text)
    text = remove_punctuation(text)
    text = to_lowercase(text)
    text = replace_numbers(text)
    words = word_tokenize(text)
    words = remove_stopwords(words)
    return ' '.join(words)



# Load tokenizer and preprocess data
@st.cache_resource
def load_tokenizer():
    tokenizer = Tokenizer()
    # You might want to load a pre-saved tokenizer state here
    return tokenizer

# Load the model
@st.cache_resource
def load_tflite_model():
    interpreter = tf.lite.Interpreter(model_path='artifacts/converted_model.tflite')
    interpreter.allocate_tensors()
    return interpreter


def main():
    st.title('Sentiment Analysis')
    
    # Load model and tokenizer
    interpreter = load_tflite_model()
    tokenizer = load_tokenizer()
    
    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Text input
    review = st.text_area('Enter your movie review')
    
    if st.button('Predict Sentiment'):
        if review:
            # Preprocess text
            processed_review = normalize_text(review)
            
            # Tokenize and pad
            tokenizer.fit_on_texts([processed_review])
            encoded_review = tokenizer.texts_to_sequences([processed_review])
            padded_review = pad_sequences(encoded_review, maxlen=100, padding='post')
            
            # Reshape to match model's expected input
            padded_review = np.tile(padded_review, (32, 1))  # 1 is batch size, 100 is sequence length
            padded_review = padded_review.astype(np.float32)
            # Set the tensor to point to the input data to be inferred
            interpreter.set_tensor(input_details[0]['index'], padded_review)
            
            # Run inference
            interpreter.invoke()
            
            # Get the output tensor
            prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]
            
            # Convert to binary sentiment
            sentiment = 'Positive' if prediction > 0.5 else 'Negative'
            confidence = prediction if prediction > 0.5 else 1 - prediction
            
            # Display results
            st.success(f'Predicted Sentiment: {sentiment}')
            st.info(f'Confidence: {confidence*100:.2f}%')

if __name__ == '__main__':
    main()
