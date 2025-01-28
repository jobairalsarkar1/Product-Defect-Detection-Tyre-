import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator

BASE_DIR = "data"
CLASSES = ["defective", "good"]
IMG_SIZE = (224, 224)
OUTPUT_SHAPE = 224


def load_and_preprocess_images(base_dir, classes, img_size):
    images, labels = [], []
    error_log = []  # Log any images that caused an error
    for label, class_name in enumerate(classes):
        class_dir = os.path.join(base_dir, class_name)
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            try:
                img = load_img(img_path, target_size=img_size)
                img_array = img_to_array(img)
                img_array = img_array / 255.0
                images.append(img_array)
                labels.append(label)
            except Exception as e:
                # Log the image path that caused an error
                error_log.append(img_path)
                print(f"Something went wrong loading image {img_path}: {e}")
    return np.array(images), np.array(labels), error_log


# Load images and labels
print("Load: Loading images and labels...")
images, labels, error_log = load_and_preprocess_images(
    BASE_DIR, CLASSES, IMG_SIZE)

# Print any errors that occurred while loading images
if error_log:
    print(
        f"\nFailed to load {len(error_log)} images. Check the following paths:\n")
    print("\n".join(error_log))

# Split the data into training, validation, and test sets
print("Splitting data...")
X_train, X_temp, y_train, y_temp = train_test_split(
    images, labels, test_size=0.3, random_state=42, stratify=labels)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Convert labels to one-hot encoded format
y_train = to_categorical(y_train, num_classes=len(CLASSES))
y_val = to_categorical(y_val, num_classes=len(CLASSES))
y_test = to_categorical(y_test, num_classes=len(CLASSES))


# Save the preprocessed data
np.savez("preprocessed_data.npz", X_train=X_train, y_train=y_train,
         X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test)

print("Process: Data processing completed.")
print(
    f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}, Test samples: {len(X_test)}")
