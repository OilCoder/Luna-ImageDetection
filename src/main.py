import os
from rename_files import rename_files
from verify_file_names import verify_file_names
from count_tool_categories import count_tool_categories
from train import train_model
from evaluate import evaluate_model
from predict import predict_all_images

def main():
    print("Luna-ImageDetection: Slickline Tool Recognition Project")
    print("======================================================")
    
    # Specify the directory where your images are stored
    image_directory = os.path.join(os.path.dirname(__file__), "..", "images")
    
    print("\nStep 1: Renaming files and creating file_names.txt")
    print("---------------------------------------------------")
    rename_files(image_directory)
    
    print("\nStep 2: Verifying file names")
    print("-----------------------------")
    verify_file_names()
    
    print("\nStep 3: Counting tool categories and listing unknown tools")
    print("-----------------------------------------------------------")
    count_tool_categories()
    
    print("\nStep 4: Training the model")
    print("--------------------------")
    train_model()
    
    print("\nStep 5: Evaluating the model")
    print("----------------------------")
    evaluate_model()
    
    print("\nStep 6: Predicting on all images")
    print("--------------------------------")
    predict_all_images()
    
    print("\nAll steps completed successfully.")
    print("\nTo run the Streamlit app, use: python src/run_app.py")

if __name__ == "__main__":
    main()