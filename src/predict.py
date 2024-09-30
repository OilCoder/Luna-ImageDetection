import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from data_preparation import load_data

def predict_image(image_path, model, cat_to_label):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.
    
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    
    label_to_cat = {v: k for k, v in cat_to_label.items()}
    predicted_category = label_to_cat[predicted_class]
    
    return predicted_category, prediction[0][predicted_class]

def predict_all_images():
    image_dir = os.path.join(os.path.dirname(__file__), "..", "images")
    labels_file = os.path.join(os.path.dirname(__file__), "..", "classified_tool_names.txt")
    
    _, _, cat_to_label = load_data(image_dir, labels_file)
    model = load_model('best_model.keras')  # Make sure this line uses .keras
    
    for filename in os.listdir(image_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            image_path = os.path.join(image_dir, filename)
            predicted_category, confidence = predict_image(image_path, model, cat_to_label)
            print(f"Image: {filename}")
            print(f"Predicted category: {predicted_category}")
            print(f"Confidence: {confidence:.2f}")
            print("---")

if __name__ == "__main__":
    predict_all_images()