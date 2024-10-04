import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
import os
import tensorflow as tf
import json
import sys

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from classic_model.data_preparation import load_data
from classic_model.model import build_model, compile_model
from classic_model.train import train_model

# Set page configuration
st.set_page_config(page_title="Oilfield Tool Classifier", layout="wide", initial_sidebar_state="expanded")

# # Custom CSS for better aesthetics
# st.markdown("""
#     <style>
#     .main {
#         background-color: #f0f2f6;
#     }
#     .stButton>button {
#         width: 100%;
#     }
#     .stImage {
#         border-radius: 10px;
#         box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
#     }
#     .stSubheader {
#         font-size: 24px;
#         font-weight: bold;
#         color: #1f77b4;
#     }
#     </style>
#     """, unsafe_allow_html=True)

@st.cache_resource
def load_model_and_labels():
    model_path = os.path.join(project_root, 'models', 'classic_model_final.keras')
    if not os.path.exists(model_path):
        st.warning("Model not found. Training a new model...")
        data = load_data()
        num_classes = len(data[2])  # Get the number of unique labels
        model = build_model(num_classes=num_classes)
        compile_model(model)
        train_data, val_data, _ = split_data(data)
        train_data, val_data, _ = preprocess_images(train_data, val_data)
        history = train_model(model, train_data, val_data)
        st.success("Model training completed.")
    else:
        model = load_model(model_path)
        
    # Load labels
    _, _, unique_labels, label_to_index = load_data()
    
    return model, unique_labels, label_to_index

def preprocess_image(image):
    image = image.convert('RGB')
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)

@st.cache_data
def load_tool_info():
    try:
        with open('tool_info.json', 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        st.error("Error loading tool information. The JSON file may be malformed.")
        return {}
    except FileNotFoundError:
        st.error("tool_info.json file not found.")
        return {}

def display_tool_info(tool_category, tool_info):
    st.markdown(f"<h3 style='color: #1f77b4;'>Information about {tool_category}</h3>", unsafe_allow_html=True)
    
    info = tool_info.get(tool_category, {})
    if info:
        st.markdown(f"**Description:** {info.get('description', 'N/A')}")
        st.markdown(f"**Common Tools:** {', '.join(info.get('common_tools', ['N/A']))}")
        st.markdown(f"**Use Case:** {info.get('use_case', 'N/A')}")
        st.markdown("**Key Features:**")
        for feature in info.get('key_features', ['N/A']):
            st.markdown(f"- {feature}")
        st.markdown(f"**Materials:** {info.get('materials', 'N/A')}")
        st.markdown(f"**Technical Specs:** {info.get('technical_specs', 'N/A')}")
        st.markdown("**Applications:**")
        for app in info.get('applications', ['N/A']):
            st.markdown(f"- {app}")
    else:
        st.write("No detailed information available for this tool category.")

def main():
    st.title("🛠️ Oilfield Tool Classifier")

    model, unique_labels, label_to_index = load_model_and_labels()
    index_to_label = {v: k for k, v in label_to_index.items()}  # Create an index_to_label dictionary
    tool_info = load_tool_info()

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns([1, 1.5])
        with col1:
            st.image(image, caption='Uploaded Image', use_column_width=True, width=300)

        preprocessed_image = preprocess_image(image)
        prediction = model.predict(preprocessed_image)
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction)
        
        predicted_category = index_to_label[predicted_class]  # Use index_to_label to get the category
        
        with col2:
            st.markdown("<h3 style='color: #1f77b4;'>Classification Result</h3>", unsafe_allow_html=True)
            st.markdown(f"**Predicted class:** {predicted_category}")
            st.markdown(f"**Confidence:** {confidence:.2f}")
            
            # Display top 3 predictions
            top_3 = np.argsort(prediction[0])[-3:][::-1]
            st.markdown("**Top 3 Predictions:**")
            for idx in top_3:
                st.markdown(f"- {index_to_label[idx]}: {prediction[0][idx]:.2f}")  # Use index_to_label to get the category

        # Display detailed information about the predicted tool
        st.markdown("---")
        display_tool_info(predicted_category, tool_info)

if __name__ == "__main__":
    main()