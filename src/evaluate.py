import os
from data_preparation import load_data
from tensorflow.keras.models import load_model

def evaluate_model():
    image_dir = os.path.join(os.path.dirname(__file__), "..", "images")
    labels_file = os.path.join(os.path.dirname(__file__), "..", "classified_tool_names.txt")
    
    _, val_generator, _ = load_data(image_dir, labels_file)
    
    model = load_model('best_model.keras')
    
    results = model.evaluate(val_generator)
    print(f"Test loss: {results[0]}")
    print(f"Test accuracy: {results[1]}")

if __name__ == "__main__":
    evaluate_model()