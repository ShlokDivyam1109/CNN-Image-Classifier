# This project is now set up for only two classes: 'yes' and 'no'.

import os
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

# Rectangle size (should match Data_Collector.py)
RECT_WIDTH = 400
RECT_HEIGHT = 300
IMG_SIZE = (200, 200)  # Should match training size
MODEL_FILE = 'model.tflite'
CLASS_NAMES_FILE = 'class_names.txt'

# Load class names
if os.path.exists(CLASS_NAMES_FILE):
	with open(CLASS_NAMES_FILE, 'r') as f:
		class_names = [line.strip() for line in f.readlines()]
else:
	class_names = None

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path=MODEL_FILE)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
	print('Cannot open camera')
	exit()

while True:
	ret, frame = cap.read()
	if not ret:
		print('Failed to grab frame')
		break

	# Flip the frame left to right
	frame = cv2.flip(frame, 1)

	h, w, _ = frame.shape
	# Center rectangle
	x1 = w // 2 - RECT_WIDTH // 2
	y1 = h // 2 - RECT_HEIGHT // 2
	x2 = x1 + RECT_WIDTH
	y2 = y1 + RECT_HEIGHT

	# Draw rectangle
	cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

	# Crop and preprocess
	crop = frame[y1:y2, x1:x2]
	img = Image.fromarray(crop).resize(IMG_SIZE).convert('RGB')
	img_np = np.array(img).astype('float32') / 255.0
	img_np = np.expand_dims(img_np, axis=0)

	# Run inference
	interpreter.set_tensor(input_details[0]['index'], img_np)
	interpreter.invoke()
	output = interpreter.get_tensor(output_details[0]['index'])[0]

	# Display probabilities with class names
	for i, prob in enumerate(output):
		if class_names and i < len(class_names):
			text = f'{class_names[i]}: {prob:.2f}'
		else:
			text = f'Class {i}: {prob:.2f}'
		cv2.putText(frame, text, (10, 60 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

	cv2.imshow('Realtime Predictor', frame)
	key = cv2.waitKey(1) & 0xFF
	if key == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
