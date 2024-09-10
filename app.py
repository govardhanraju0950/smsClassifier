import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image, UnidentifiedImageError

# Load the trained model
model = load_model('cat_dog_classifier.h5')

# Streamlit app
st.title("Cat and Dog Classifier")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Attempt to open the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Classifying...")

        # Preprocess the image
        image = image.resize((150, 150))  # Resize the image to 150x150
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image /= 255.0

        # Make prediction
        prediction = model.predict(image)
        result = "Dog" if prediction[0][0] > 0.5 else "Cat"
        st.write(f"Prediction: {result}")

    except UnidentifiedImageError:
        st.error("The uploaded file is not a valid image. Please upload a JPG, JPEG, or PNG file.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
else:
    st.write("Please upload an image file.")
