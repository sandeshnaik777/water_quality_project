# test_image.py
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# Load the saved model
model_path = 'water_quality_model.h5'
model = load_model(model_path)
print("Model loaded successfully!")

# Image settings
img_height, img_width = 150, 150

# Open file dialog to select image
Tk().withdraw()  # Hide main window
file_path = askopenfilename(title="Select an image for testing")

if file_path:
    img = image.load_img(file_path, target_size=(img_height, img_width))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)[0][0]

    if prediction < 0.5:
        print("Prediction: CLEAN WATER")
    else:
        print("Prediction: DIRTY WATER")
else:
    print("No image selected.")
