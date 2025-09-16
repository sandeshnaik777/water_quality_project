# 💧 Water Quality Classification 🤖

Hey there! Welcome to the Water Quality Classification project. This is a cool little AI model that uses a Convolutional Neural Network (CNN) to tell if water in an image is clean or dirty. Perfect for environmental monitoring or just satisfying your curiosity about water quality!

## 🌟 What This Project Does
- **Classifies water images**: Upload a photo of water, and the model predicts if it's clean or dirty.
- **Built with TensorFlow**: Uses deep learning to analyze images like a pro.
- **Easy to use**: Simple scripts for training your own model or testing with existing ones.

## 🚀 Features
- 🧠 CNN model with data augmentation for better accuracy
- 📊 Binary classification: Clean vs. Dirty water
- 🖼️ Works with JPG images
- 🖥️ User-friendly scripts for training and testing
- 📁 Organized dataset structure

## 📋 Requirements
You'll need these to run the project:
- Python 3.x
- TensorFlow (the brain behind the AI)
- NumPy (for number crunching)
- Tkinter (for the file picker when testing)

## 🛠️ Installation
1. Grab the project files from this repo.
2. Install the goodies:
   ```
   pip install tensorflow numpy
   ```
   (Tkinter usually comes with Python – if not, look it up for your system!)

## 📂 Dataset Setup
Your images should be sorted like this:
```
datasets/
├── train/
│   ├── clean/     # Put clean water images here
│   └── dirty/     # Dirty water images go here
└── test/
    ├── clean/     # Test clean images
    └── dirty/     # Test dirty images
```
Each folder has JPGs of the respective water types.

## 🎯 How to Use

### Train Your Own Model
Want to train the model yourself? Run:
```
python train_model.py
```
It'll train for 100 epochs, learn from your data, and save the model as `water_quality_model.h5`. Sit back and watch the magic happen!

### Test the Model
Ready to classify some water? Run:
```
python test_image.py
```
A window pops up – pick an image, and boom! It tells you if it's clean or dirty water.

### Download and Use the Pre-Trained Model
If you just want to test without training:
1. Download `water_quality_model.h5` from the releases or wherever it's hosted.
2. Place it in the project root.
3. Run `test_image.py` as above – it's ready to go!

## 🏗️ Model Details
The CNN has:
- 3 convolutional layers with ReLU and max pooling
- A flatten layer
- Dense layers with dropout to avoid overfitting
- Sigmoid output for yes/no clean/dirty

## 📜 License
This is open source under the MIT License. Feel free to tweak and share!

Got questions? Open an issue or reach out. Happy classifying! 🌊
