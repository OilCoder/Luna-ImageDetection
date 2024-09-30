import os
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from data_preparation import load_data

class ToolRecognitionApp:
    def __init__(self, master):
        self.master = master
        master.title("Slickline Tool Recognition")

        self.model = load_model('best_model.keras')
        
        image_dir = os.path.join(os.path.dirname(__file__), "..", "images")
        labels_file = os.path.join(os.path.dirname(__file__), "..", "classified_tool_names.txt")
        _, _, self.cat_to_label = load_data(image_dir, labels_file)
        self.label_to_cat = {v: k for k, v in self.cat_to_label.items()}

        self.load_button = tk.Button(master, text="Load Image", command=self.load_image)
        self.load_button.pack()

        self.image_label = tk.Label(master)
        self.image_label.pack()

        self.result_label = tk.Label(master, text="")
        self.result_label.pack()

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            image = Image.open(file_path)
            image = image.resize((300, 300))  # Resize for display
            photo = ImageTk.PhotoImage(image)
            self.image_label.config(image=photo)
            self.image_label.image = photo

            # Predict
            predicted_category, confidence = self.predict_image(file_path)
            self.result_label.config(text=f"Predicted category: {predicted_category}\nConfidence: {confidence:.2f}")

    def predict_image(self, image_path):
        img = Image.open(image_path).resize((224, 224))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.

        prediction = self.model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)[0]
        predicted_category = self.label_to_cat[predicted_class]
        
        return predicted_category, prediction[0][predicted_class]

if __name__ == "__main__":
    root = tk.Tk()
    app = ToolRecognitionApp(root)
    root.mainloop()