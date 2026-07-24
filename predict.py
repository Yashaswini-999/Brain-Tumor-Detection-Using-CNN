import cv2
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model

# Load trained model
model = load_model("brain_tumor_model.keras")

# Path of test image
image_path = "test.jpg"

# Read image
image = cv2.imread(image_path)

# Check if image exists
if image is None:
    print("Image not found!")
    exit()

# Convert BGR to RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Resize image
image = cv2.resize(image, (128,128))

# Normalize
image = image / 255.0

# Add batch dimension
image = np.expand_dims(image, axis=0)

# Predict
prediction = model.predict(image)

if prediction[0][0] > 0.5:
    result = "Tumor Detected"
else:
    result = "No Tumor"

# Display
plt.imshow(image_rgb)
plt.title(result)
plt.axis("off")
plt.savefig("sample_prediction.png")
plt.show()

print("Prediction Score:", prediction[0][0])
print("Result:", result)