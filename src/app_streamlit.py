import streamlit as st
import os
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from data_preparation import load_data

def load_model_and_labels():
    model = load_model('best_model.keras')
    
    image_dir = os.path.join(os.path.dirname(__file__), "..", "images")
    labels_file = os.path.join(os.path.dirname(__file__), "..", "classified_tool_names.txt")
    _, _, cat_to_label = load_data(image_dir, labels_file)
    label_to_cat = {v: k for k, v in cat_to_label.items()}
    
    return model, label_to_cat

def predict_image(image, model, label_to_cat):
    img_array = img_to_array(image.resize((224, 224)))
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.
    
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    
    predicted_category = label_to_cat[predicted_class]
    confidence = prediction[0][predicted_class]
    
    return predicted_category, confidence

def main():
    st.title("Slickline Tool Recognition")

    model, label_to_cat = load_model_and_labels()

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Classifying...")
        predicted_category, confidence = predict_image(image, model, label_to_cat)
        st.write(f"Predicted category: {predicted_category}")
        st.write(f"Confidence: {confidence:.2f}")

if __name__ == "__main__":
    main()