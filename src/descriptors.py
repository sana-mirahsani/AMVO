import numpy as np
import os
import cv2
from skimage.feature import local_binary_pattern

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

def calculate_color_histogram_all(path_file_label, path_file_images, func_histogram, **kwargs):
	
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

		if func_histogram.__name__ == 'color_histogram' or func_histogram.__name__ == 'geometric_segmentation_of_images':
			im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
		
		hist = func_histogram(im, **kwargs)

		X.append(hist)
		y.append(label)

	X = np.array(X)
	y = np.array(y)

	return X, y

def lbp_histogram(im, P=8, R=1, method='uniform'):
    """
    Computes LBP histogram for a single image.

    Parameters:
    -----------
    im : numpy array, shape (H, W, 3)
        Color image (RGB).
    P : int
        Number of points for LBP.
    R : int
        Radius for LBP.
    method : str
        LBP method ('uniform' is standard).

    Returns:
    --------
    hist : numpy array, shape (P*(P-1)+3,)
        Normalized LBP histogram
    """
    # Convert to grayscale
    gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)

    # Compute LBP image
    lbp = local_binary_pattern(gray, P, R, method=method)

    # Compute histogram of LBP codes
    n_bins = int(lbp.max() + 1)  # number of unique LBP codes
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))

    # Normalize histogram
    hist = hist.astype(np.float32)
    hist /= hist.sum()

    return hist

def merging_descriptors(X_desc1, X_desc2):
	X_fused = np.hstack((X_desc1, X_desc2))
	return X_fused

def geometric_segmentation_of_images(img, grid, bins):
	# calculate histogram for ONE image
	H, W = img.shape[:2]

	h_step = H // grid
	w_step = W // grid

	# cut image in regions
	regions = [
		img[i*h_step:(i+1)*h_step, j*w_step:(j+1)*w_step]
		for i in range(grid)
		for j in range(grid)
	]

	# calculate the histogram for all regions
	features = []

	# convert list to array
	regions_array = np.array(regions)

	# regions_array has shape (25, region_h, region_w, 3)
	for region in regions_array:
		# descriptor
		h = color_histogram(region, bins)  
		features.append(h)

	# Final feature vector for this image
	final_descriptor = np.concatenate(features)

	return final_descriptor