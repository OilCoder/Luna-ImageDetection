import os
from data_preparation import load_data
from model import create_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

def train_model():
    image_dir = os.path.join(os.path.dirname(__file__), "..", "images")
    labels_file = os.path.join(os.path.dirname(__file__), "..", "classified_tool_names.txt")
    
    train_generator, val_generator, cat_to_label = load_data(image_dir, labels_file)
    
    model = create_model(len(cat_to_label))
    
    checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_accuracy', mode='max')
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=20,
        validation_data=val_generator,
        validation_steps=len(val_generator),
        callbacks=[checkpoint, early_stop]
    )
    
    return history, model

if __name__ == "__main__":
    history, model = train_model()
    print("Training completed.")