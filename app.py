import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np

model = load_model("model.keras")
class_names = ["Bacterial spot", "Early blight", "Healthy", "Late blight", "Leaf mold",
               "Mosaic virus", "Septoria leaf spot", "Spider mite", "Target spot", "Yellowleaf curl virus"]


def preprocess_image(image):
    image_arr = img_to_array(image)
    image_arr = tf.expand_dims(image_arr, 0)
    return image_arr


def predict_image_class(image):
    image_arr = preprocess_image(image)
    prediction = model.predict(image_arr)
    predicted_class = class_names[np.argmax(prediction[0])]
    confidence = np.round(100*(np.max(prediction[0])), 2)
    return predicted_class, confidence


st.title("Tomato plant disease classification:")

uploaded_file = st.file_uploader("Choose a file:", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded image', use_column_width=True)
    predicted_class, confidence = predict_image_class(image)
    st.write("Results:\n")
    st.write(f"Predicted disease: {predicted_class}")
    st.write(f"Confidence: {confidence}%")
