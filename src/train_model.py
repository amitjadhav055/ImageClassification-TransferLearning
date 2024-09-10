# src/train_model.py

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from build_model import build_model

def train_model():
    # Paths to your dataset
    train_dir = 'data/train'
    test_dir = 'data/test'

    # Data augmentation and preprocessing
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary'
    )

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary'
    )

    # Build and compile the model
    input_shape = (150, 150, 3)
    num_classes = 2
    model = build_model(input_shape, num_classes)

    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // 32,
        epochs=10,
        validation_data=test_generator,
        validation_steps=test_generator.samples // 32
    )

    # Save the model
    model.save('models/cats_dogs_model.h5')

if __name__ == "__main__":
    train_model()
