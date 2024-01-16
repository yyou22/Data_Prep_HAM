import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import pandas as pd

def convert_img():

	path = "./Images_resize/"
	if not os.path.exists(path):
		os.makedirs(path)

	images = np.load('X.npy')

	num = np.shape(images)[0]

	for i in range(0, num):

		img = np.transpose(images[i], (1, 2, 0))
		img = (img*255).astype(np.uint8)

		#plt.imshow(img)
		#plt.show()

		s1 = f'{i:05d}'

		img_ = Image.fromarray(img, 'RGB')
		img_.save(path + s1 + '.jpg')

		#x = np.shape(img)[0]
		#y = np.shape(img)[1]

		#plt.imshow(img)
		#plt.show()

		#s1 = f'{i:05d}'

		#plt.savefig(s1 + '.jpg')

def combine_csv():

	a = pd.read_csv("inv.csv")
	b = pd.read_csv("summary.csv")
	merged = pd.concat((b, a),axis=1)
	merged.to_csv("output.csv", index=False)

def main():
	#convert_img()
	combine_csv()

if __name__ == '__main__':
	main()