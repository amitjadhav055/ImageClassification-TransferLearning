import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define directories
train_dir = os.path.join('data', 'train')
test_dir = os.path.join('data', 'test')

# Data augmentation and rescaling
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

# Load and preprocess training data
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),  # Resize images
    batch_size=20,
    class_mode='binary'  # Binary classification (cats vs dogs)
)

# Load and preprocess validation data
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary'
)

if __name__ == "__main__":
    # Test the generators
    print("Training classes: ", train_generator.class_indices)
    print("Test classes: ", test_generator.class_indices)
