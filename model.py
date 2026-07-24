import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

# Dataset path
dataset_path = "dataset"

images = []
labels = []

# Load images
for folder in ["yes", "no"]:

    folder_path = os.path.join(dataset_path, folder)

    for file in os.listdir(folder_path):

        image_path = os.path.join(folder_path, file)

        image = cv2.imread(image_path)

        if image is None:
            continue

        image = cv2.resize(image, (128, 128))

        images.append(image)

        if folder == "yes":
            labels.append(1)      # Tumor
        else:
            labels.append(0)      # No Tumor

# Convert to numpy
X = np.array(images, dtype="float32") / 255.0
y = np.array(labels)

print("Total Images:", len(X))
print("Image Shape:", X.shape)
print("Labels Shape:", y.shape)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training Images:", X_train.shape)
print("Testing Images:", X_test.shape)

# CNN Model
model = Sequential()

model.add(Conv2D(32, (3,3), activation="relu", input_shape=(128,128,3)))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(64, (3,3), activation="relu"))
model.add(MaxPooling2D(2,2))

model.add(Flatten())

model.add(Dense(128, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Train
history = model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test, y_test)
)

# Evaluate
loss, accuracy = model.evaluate(X_test, y_test)

print("\nTest Accuracy:", accuracy)

# Save model
model.save("brain_tumor_model.keras")

print("Model Saved Successfully!")

# Accuracy Graph
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label="Training Accuracy")
plt.plot(history.history['val_accuracy'], label="Validation Accuracy")
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label="Training Loss")
plt.plot(history.history['val_loss'], label="Validation Loss")
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()

plt.savefig("accuracy_loss_graph.png")

plt.close()
print("Model Saved Successfully!")
print("Training Completed!")
print("You can now run: python predict.py")