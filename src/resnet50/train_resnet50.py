import os
from data_preparation import load_data
from model import create_classification_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

def train_model():
    image_dir = os.path.join(os.path.dirname(__file__), "..", "images")
    labels_file = os.path.join(os.path.dirname(__file__), "..", "classified_tool_names.txt")
    
    train_generator, val_generator, cat_to_label = load_data(image_dir, labels_file)
    
    if len(train_generator.class_indices) == 0:
        raise ValueError("No images found in the specified directory. Please check your image_dir path.")
    
    num_classes = len(cat_to_label)
    model = create_classification_model(num_classes)
    
    checkpoint = ModelCheckpoint('best_classification_model.keras', save_best_only=True, monitor='val_accuracy', mode='max')
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
    
    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=30,
        validation_data=val_generator,
        validation_steps=len(val_generator),
        callbacks=[checkpoint, early_stop, reduce_lr]
    )
    
    return history, model, cat_to_label

if __name__ == "__main__":
    try:
        history, model, cat_to_label = train_model()
        print("Training completed.")
        print(f"Number of classes: {len(cat_to_label)}")
        print(f"Classes: {cat_to_label}")
    except ValueError as e:
        print(f"Error: {e}")
        print("Please ensure that your image directory contains subdirectories for each class, and that the labels file matches the subdirectory names.")