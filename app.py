from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Initialize app
app = Flask(__name__)

# Load trained model
model = load_model('water_quality_model.h5')
img_height, img_width = 150, 150

# Folder to store uploaded images
UPLOAD_FOLDER = 'static/uploads/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_text = ""
    img_path = None

    if request.method == 'POST':
        file = request.files['file']
        if file:
            img_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(img_path)

            # Preprocess image
            img_array = image.load_img(img_path, target_size=(img_height, img_width))
            img_array = image.img_to_array(img_array)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            # Predict
            prediction = model.predict(img_array)[0][0]
            prediction_text = "CLEAN WATER" if prediction < 0.5 else "DIRTY WATER"

    return render_template('index.html', prediction=prediction_text, img_path=img_path)

if __name__ == '__main__':
    app.run(debug=True)
