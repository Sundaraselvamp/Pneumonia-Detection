import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.vgg16 import preprocess_input

# Load the Saved model
model = load_model('notebook\object_detection_model.h5')

# Define the classes
classes = ['Not Pneumonia Affected', 'Pneumonia Affected']

# Function to preprocess the uploaded image
def preprocess_image(image_data):
    img = Image.open(image_data)
    
    # Convert grayscale to RGB if necessary
    if img.mode != "RGB":
        img = img.convert("RGB")
    
    img = img.resize((224, 224))
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    processed_img = preprocess_input(img_array)
    return processed_img

# Streamlit app
st.title('Pneumonia Detection App')

# Upload image
uploaded_file = st.file_uploader("Choose the image...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    preprocessed_image = preprocess_image(uploaded_file)

    # Make prediction
    prediction = model.predict(preprocessed_image)
    predicted_class = classes[np.argmax(prediction)]

    # Display prediction
    st.write(f'Prediction: {predicted_class}')
