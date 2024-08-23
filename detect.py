import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# Load the trained model
model = load_model('breast_cancer_classifier.h5')

# Function to load and preprocess the image
def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))  # Adjust target_size to match your model's input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Rescale the image (same as during training)
    return img_array

# Path to the image you want to predict
img_path = 'path_to_image' 

# Preprocess the image
img_array = load_and_preprocess_image(img_path)

# Perform inference
prediction = model.predict(img_array)

# Output the result
if prediction[0] > 0.5:
    print(f'The image is predicted to be malignant with a confidence of {prediction[0][0]:.4f}')
else:
    print(f'The image is predicted to be benign with a confidence of {1 - prediction[0][0]:.4f}')

# Display the image along with prediction
img = image.load_img(img_path, target_size=(150, 150))
plt.imshow(img)
plt.axis('off')
plt.title(f'Predicted: {"Malignant" if prediction[0] > 0.5 else "Benign"}')
plt.show()
