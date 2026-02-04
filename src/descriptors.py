import numpy as np

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