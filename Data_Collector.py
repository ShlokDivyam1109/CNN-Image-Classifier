# import required libraries
import cv2
import os

# Rectangle size (change as needed)
RECT_WIDTH = 400
RECT_HEIGHT = 300

# Output directory
OUTPUT_DIR = './Image_Dataset/'
if not os.path.exists(OUTPUT_DIR):
	os.makedirs(OUTPUT_DIR)

# Find the next available image number
def get_next_image_number():
	files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.png')]
	nums = [int(os.path.splitext(f)[0]) for f in files if os.path.splitext(f)[0].isdigit()]
	return max(nums, default=0) + 1

cap = cv2.VideoCapture(0)
if not cap.isOpened():
	print('Cannot open camera')
	exit()

save_images = False
img_count = get_next_image_number()

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
	cv2.putText(frame, f'Saving: {save_images}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255) if save_images else (255,0,0), 2)
	cv2.imshow('Camera', frame)

	key = cv2.waitKey(1) & 0xFF
	if key == ord('q'):
		break
	elif key == ord('s'):
		save_images = not save_images
		print(f'Last Saved image: {img_count}')
	if save_images:
		crop = frame[y1:y2, x1:x2]
		cv2.imwrite(os.path.join(OUTPUT_DIR, f'{img_count}.png'), crop)
		img_count += 1

cap.release()
cv2.destroyAllWindows()
