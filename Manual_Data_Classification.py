import os

IMAGE_DIR = './Image_Dataset/'
CSV_FILE = 'Data.csv'

def get_sorted_image_files():
	files = [f for f in os.listdir(IMAGE_DIR) if f.endswith('.png') and f[:-4].isdigit()]
	files = sorted(files, key=lambda x: int(os.path.splitext(x)[0]))
	return files

def main():
	images = get_sorted_image_files()
	if not images:
		print('No images found in', IMAGE_DIR)
		return

	print(f'Total images: {len(images)}')
	print('You will be asked to enter the start and end image numbers for each class.')
	print('For example, if you want to classify images 1.png to 10.png as class yes, enter 1 10 and then yes.')
	print('Only two classes are allowed: yes or no.')
	print('Enter "done" when finished.')

	class_ranges = []
	used = set()
	class_id = 0
	while True:
		inp = input(f'Enter start and end image numbers for class {class_id} (or "done"): ')
		if inp.strip().lower() == 'done':
			break
		try:
			start, end = map(int, inp.strip().split())
		except Exception:
			print('Invalid input. Please enter two numbers or "done".')
			continue
		if start > end:
			print('Start must be <= end.')
			continue
		# Mark used images
		while True:
			cls = input(f'Enter class name for the above range (yes/no): ').strip().lower()
			if cls not in ["yes", "no"]:
				print('Invalid class name. Please enter "yes" or "no".')
			else:
				break
		for i in range(start, end+1):
			if i in used:
				print(f'Image {i}.png already classified. Skipping.')
				continue
			class_ranges.append((f'{IMAGE_DIR}{i}.png', cls))
			used.add(i)
		class_id += 1

	# Write to CSV
	with open(CSV_FILE, 'w') as f:
		f.write('image,class\n')
		for img, cls in class_ranges:
			f.write(f'{img},{cls}\n')
	print(f'Classification saved to {CSV_FILE}')

if __name__ == '__main__':
	main()
