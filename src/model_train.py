import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load preprocessed data
data = np.load("preprocessed_data.npz")
X_train, y_train = data["X_train"], data["y_train"]
X_val, y_val = data["X_val"], data["y_val"]

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
datagen.fit(X_train)

# Define model architecture


def build_model(input_shape, num_classes):
    base_model = EfficientNetB0(
        include_top=False, weights="imagenet", input_shape=input_shape)
    base_model.trainable = False  # Freeze base model during initial training

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.4)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    output = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=output)
    return model


# Build the model
input_shape = (224, 224, 3)
num_classes = 2
model = build_model(input_shape, num_classes)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

# Define callbacks
checkpoint = ModelCheckpoint("saved_models/best_model.h5",
                             monitor="val_accuracy", save_best_only=True, verbose=1)
early_stopping = EarlyStopping(
    monitor="val_accuracy", patience=5, restore_best_weights=True)

# Train the model
batch_size = 32
epochs = 20  # Increase the number of epochs for better performance

history = model.fit(
    datagen.flow(X_train, y_train, batch_size=batch_size),
    validation_data=(X_val, y_val),
    epochs=epochs,
    callbacks=[checkpoint, early_stopping]
)

# Save the final model
model.save("saved_models/final_model.h5")
print("Model training successfully completed. Final model saved to saved_models/final_model.h5.")
