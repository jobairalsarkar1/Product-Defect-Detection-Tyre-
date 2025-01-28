import os
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Function to preprocess a single image


def preprocess_image(image_path, img_size=(224, 224)):
    img = load_img(image_path, target_size=img_size)
    img_array = img_to_array(img)
    # Expand dimensions to create a batch of 1 image
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize pixel values
    return img_array


# Load the trained model
model = tf.keras.models.load_model("saved_models/best_model.h5")

# Function to make a prediction on a single image


def make_prediction(image_path):
    if not os.path.isfile(image_path):
        print("Error: The file path is invalid or the file does not exist.")
        return

    img_array = preprocess_image(image_path)

    # Make a prediction (0 for defective, 1 for good)
    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions, axis=1)[0]

    # Map the class index to a class label
    class_labels = ["Defective", "Good"]
    predicted_label = class_labels[class_idx]

    # Display the image and the prediction
    img = load_img(image_path)
    plt.imshow(img)
    plt.axis("off")
    plt.title(
        f"Prediction: {predicted_label} (Confidence: {predictions[0][class_idx]:.2f})")
    plt.show()

    return predicted_label


# Parse command-line arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict tyre defect status")
    parser.add_argument('image_path', type=str,
                        help="Path to the image file for prediction")

    args = parser.parse_args()

    # Make a prediction
    predicted_class = make_prediction(args.image_path)
    print(f"The tyre is predicted to be: {predicted_class}")
