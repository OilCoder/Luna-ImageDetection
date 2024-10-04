import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from PIL import Image
import cv2
from data_preparation import load_data

# Set the models directory
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
IMAGE_DIR = os.path.join(os.path.dirname(__file__), "..", "images")
LABELS_FILE = os.path.join(os.path.dirname(__file__), "..", "classified_tool_names.txt")

# ResNet50 Classification Model
class ClassificationModel:
    def __init__(self, num_classes):
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        x = GlobalAveragePooling2D()(base_model.output)
        x = Dense(1024, activation='relu')(x)
        output = Dense(num_classes, activation='softmax')(x)
        self.model = Model(inputs=base_model.input, outputs=output)
        # Load pre-trained weights
        weights_path = os.path.join(MODELS_DIR, "resnet50_classification_weights.h5")
        if os.path.exists(weights_path):
            self.model.load_weights(weights_path)
            print(f"Loaded classification weights from {weights_path}")
        else:
            print(f"Warning: Classification weights not found at {weights_path}. Using untrained model.")

    def classify_tool(self, image):
        image = cv2.resize(image, (224, 224))
        preprocessed = preprocess_input(image)
        prediction = self.model.predict(np.expand_dims(preprocessed, axis=0))
        return prediction

# Main OilfieldToolClassifier
class OilfieldToolClassifier:
    def __init__(self):
        _, _, self.cat_to_label = load_data(IMAGE_DIR, LABELS_FILE)
        num_classes = len(self.cat_to_label)
        self.classification_model = ClassificationModel(num_classes)
        self.label_to_cat = {v: k for k, v in self.cat_to_label.items()}

    def classify(self, image):
        classification_result = self.classification_model.classify_tool(image)
        return classification_result

# Usage example
if __name__ == "__main__":
    classifier = OilfieldToolClassifier()
    
    # Load and preprocess your image here
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if image_files:
        image_path = os.path.join(IMAGE_DIR, image_files[0])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Classify the image
        result = classifier.classify(image)
        
        # Process the result
        predicted_index = np.argmax(result)
        predicted_category = classifier.label_to_cat[predicted_index]
        print(f"Predicted tool category: {predicted_category}")
    else:
        print(f"Error: No image files found in {IMAGE_DIR}")