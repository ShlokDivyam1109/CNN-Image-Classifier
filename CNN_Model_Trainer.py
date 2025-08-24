import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from PIL import Image

CSV_FILE = 'Data.csv'
MODEL_FILE = 'model.tflite'
IMG_SIZE = (200, 200)  # You can change this to match your rectangle size

def load_data(csv_file, img_size):
	df = pd.read_csv(csv_file)
	images = []
	labels = []
	for idx, row in df.iterrows():
		img_path = row['image']
		label = row['class']
		if not os.path.exists(img_path):
			continue
		img = Image.open(img_path).convert('RGB').resize(img_size)
		images.append(np.array(img))
		labels.append(label)
	return np.array(images), np.array(labels)

def build_model(num_classes, input_shape):
	model = keras.Sequential([
		layers.Input(shape=input_shape),
		layers.Conv2D(32, (3,3), activation='relu'),
		layers.MaxPooling2D(2,2),
		layers.Conv2D(64, (3,3), activation='relu'),
		layers.MaxPooling2D(2,2),
		layers.Flatten(),
		layers.Conv2D(64, (3,3), activation='relu'),
		layers.MaxPooling2D(2,2),
		layers.Flatten(),
		layers.Dense(128, activation='relu'),
		layers.Dense(num_classes, activation='softmax')
	])
	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
	return model

def main():
	# Load data
	images, labels = load_data(CSV_FILE, IMG_SIZE)
	if len(images) == 0:
		print('No images found. Exiting.')
		return
	images = images.astype('float32') / 255.0

	# Encode labels
	lb = LabelBinarizer()
	labels_bin = lb.fit_transform(labels)
	if labels_bin.shape[1] == 1:
		labels_bin = np.hstack([1-labels_bin, labels_bin])

	# Save class names for use in predictor
	class_names = lb.classes_.tolist()
	with open('class_names.txt', 'w') as f:
		for name in class_names:
			f.write(f'{name}\n')

	# Train/test split
	X_train, X_test, y_train, y_test = train_test_split(images, labels_bin, test_size=0.2, random_state=42)

	# Build and train model
	model = build_model(num_classes=labels_bin.shape[1], input_shape=(*IMG_SIZE, 3))
	model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

	# Save as TFLite
	converter = tf.lite.TFLiteConverter.from_keras_model(model)
	tflite_model = converter.convert()
	with open(MODEL_FILE, 'wb') as f:
		f.write(tflite_model)
	print(f'Model saved as {MODEL_FILE}')

if __name__ == '__main__':
	main()
