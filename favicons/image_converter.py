from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def avg_pixel(pixel):
	return 0.299 * pixel[0] + 0.587 * pixel[1] + 0.114 * pixel[2]

def getImageGraysacleArray(image_path, size_row, size_col):
	image = misc.imread(image_path)
	image_scaled = misc.imresize(image, (size_row, size_col))
	# image_scaled = image

	grey = np.zeros((image_scaled.shape[0], image_scaled.shape[1]))
	print grey.shape

	for row in range(len(image_scaled)):
		for col in range(len(image_scaled[row])):
			grey[row][col] = avg_pixel(image_scaled[row][col])

	return grey.flatten()

if __name__ == '__main__':
	grey = getImageGraysacleArray("input_data/facebook/facebook.png", 16, 16)
	plt.imshow(grey, cmap = cm.Greys_r)
	plt.show()