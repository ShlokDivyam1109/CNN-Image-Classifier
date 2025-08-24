# CNN Image Classifier Toolkit

## Overview
This project provides a complete pipeline for collecting, labeling, training, and real-time prediction of images using a Convolutional Neural Network (CNN) with TensorFlow/Keras. It includes:
- Data collection from webcam
- Manual data classification
- Model training and export to TFLite
- Real-time prediction using the trained model

## Requirements

**Python Version:**
- Python 3.10, 3.11, or 3.12 (TensorFlow does not support Python 3.13 as of August 2025)

**Required Packages:**
- tensorflow >=2.12
- keras >=2.12
- pandas
- scikit-learn
- pillow
- opencv-python

Install all requirements with:
```sh
pip install tensorflow keras pandas scikit-learn pillow opencv-python
```

## Usage

### 1. Collect Images
Run the data collector to capture images from your webcam:
```sh
python Data_Collector.py
```
- Press `s` to start/stop saving images (cropped to a rectangle in the center).
- Press `q` to quit.
- Images are saved in `./Image_Dataset/` as 1.png, 2.png, ...

### 2. Manually Classify Images
Label the collected images with your own class names:
```sh
python Manual_Data_Classification.py
```
- Enter the start and end image numbers for each class range.
- Enter the class name (yes/no) for each range.
- Repeat for all classes, type `done` when finished.
- This creates/overwrites `Data.csv` with `image,class` columns.

### 3. Train the CNN Model
Train the model and export it as a TFLite file:
```sh
python CNN_Model_Trainer.py
```
- This reads `Data.csv`, trains a CNN, and saves `model.tflite` and `class_names.txt`.

### 4. Real-Time Prediction
Run the real-time predictor to see class probabilities from the webcam:
```sh
python Realtime_Predictor.py
```
- The model will display the probability for each class (using yes/no) in real time.
- Press `q` to quit.

## Notes
- Make sure your Python version is compatible (3.10, 3.11, or 3.12).
- If you encounter errors with TensorFlow/Keras, check your Python version and package versions.
- The rectangle size and image size can be adjusted in the scripts as needed.

## File Descriptions
- `Data_Collector.py`: Collects and saves cropped images from webcam.
- `Manual_Data_Classification.py`: Assigns class names to image ranges and creates `Data.csv`.
- `CNN_Model_Trainer.py`: Trains a CNN and exports a TFLite model and class names.
- `Realtime_Predictor.py`: Uses the TFLite model for real-time webcam prediction.

---
Created August 2025
