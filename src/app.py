import os
import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import sys

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from classic_model.data_preparation import load_data, split_data, preprocess_images
from classic_model.model import build_model, compile_model
from classic_model.train import train_model

class ToolRecognitionApp:
    def __init__(self):
        self.model_path = os.path.join(project_root, 'models', 'classic_model_final.keras')
        self.image_dir = os.path.join(project_root, "images")
        self.labels_file = os.path.join(project_root, "classified_tool_names.txt")

        self.train_generator, self.val_generator, _, self.cat_to_label, self.train_steps, self.val_steps, _ = load_and_prepare_data(self.image_dir, self.labels_file)
        self.label_to_cat = {v: k for k, v in self.cat_to_label.items()}
        self.num_classes = len(self.cat_to_label)

        if os.path.exists(self.model_path):
            self.model = load_model(self.model_path)
        else:
            st.warning("Model file not found. Please train the model first.")
            self.model = None

    def train_and_validate(self):
        st.info("Training the model...")
        history, self.model = train_model(self.train_generator, self.val_generator, self.num_classes, self.model_path, self.train_steps, self.val_steps)
        st.success("Training completed.")

        st.info("Validating the model...")
        val_results = self.validate_model()
        self.display_validation_results(val_results)

    def validate_model(self):
        results = []
        for i in range(self.val_steps):
            x, y_true = next(self.val_generator)
            y_pred = self.model.predict(x)
            for j in range(len(x)):
                true_class = np.argmax(y_true[j])
                pred_class = np.argmax(y_pred[j])
                results.append({
                    'File': f'val_image_{i*32+j}',  # Assuming batch size of 32
                    'True Label': self.label_to_cat[true_class],
                    'Predicted Label': self.label_to_cat[pred_class],
                    'Correct': true_class == pred_class
                })
        return results

    def display_validation_results(self, results):
        df = pd.DataFrame(results)
        st.write("Validation Results:")
        st.dataframe(df)

        accuracy = df['Correct'].mean()
        st.write(f"Validation Accuracy: {accuracy:.2%}")

        # Create a bar chart of prediction distribution
        pred_dist = df['Predicted Label'].value_counts()
        fig, ax = plt.subplots()
        pred_dist.plot(kind='bar', ax=ax)
        plt.title("Distribution of Predicted Labels")
        plt.xlabel("Category")
        plt.ylabel("Count")
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig)

    def predict_image(self, image):
        img_array = img_to_array(image.resize((224, 224)))
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.

        prediction = self.model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)[0]
        predicted_category = self.label_to_cat[predicted_class]
        
        return predicted_category, prediction[0][predicted_class]

def main():
    st.title("Slickline Tool Recognition")

    app = ToolRecognitionApp()

    if st.button("Train and Validate Model"):
        app.train_and_validate()

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)

        if app.model is not None:
            predicted_category, confidence = app.predict_image(image)
            st.write(f"Predicted category: {predicted_category}")
            st.write(f"Confidence: {confidence:.2f}")
        else:
            st.write("Model not loaded. Unable to make predictions.")

if __name__ == "__main__":
    main()