import numpy as np
import os
import cv2

def color_histogram(im, bins_per_channel=8):
	''' Computes a joint color histogram.
	:param im Color image as a Numpy array of shape (height, width, 3)
	@param bins_per_channel Number of bins per channel after quantization
	@type im Numpy array of type uint8 and shape (height, width, 3)
	@type bins_per_channel Integer
	@return Normalized color histogram
	@rtype Numpy array of type float32 and shape (bins_per_channel**3,)
	'''
	im = im.copy()

	# quantize image
	bin_width = 256./bins_per_channel
	im = (im / bin_width).astype(np.uint32)

	# flatten color space
	im = im[...,0] * bins_per_channel**2 + im[...,1] * bins_per_channel + im[...,2]

	# compute and normalize histogram
	histogram = np.zeros((bins_per_channel**3,), dtype=np.float32)
	colors, counts = np.unique(im, return_counts=True)
	histogram[colors] = counts
	histogram = histogram / np.linalg.norm(histogram, ord=1)
	return histogram

def calculate_color_histogram_all(path_file_label,path_file_images):
	
	labels = {} # dict of labels
	X = [] # histogram
	y = [] # labels

	with open(path_file_label, "r") as f:
		for line in f:
			image_name, label = line.strip().split()
			image_name = image_name.split('/')[1]
			labels[image_name] = label


	for image_name, label in labels.items():
		image_path = os.path.join(path_file_images, image_name)

		im = cv2.imread(image_path)
		if im is None:
			raise FileNotFoundError(f"Could not read image: {image_path}") 

		im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
		hist = color_histogram(im, 8)

		X.append(hist)
		y.append(label)

	X = np.array(X)
	y = np.array(y)

	return X, y