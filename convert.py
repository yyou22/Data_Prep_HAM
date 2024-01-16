import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import argparse
import pandas as pd
import torch

parser = argparse.ArgumentParser(description='Converting images from tensor for HAM10000 dataset')
parser.add_argument('--saved_file_path', type=str, default='./checkpoints/pixel_vgg_1103_5000_96_0.2000_rand_standard.pth', help='Path to the saved adversarial images')
parser.add_argument('--image_file_path', type=str, default='./checkpoints/images_vgg_1103_96.pth', help='Path to the sampled images and labels')

args = parser.parse_args()

def find_icons():

	saved_data = torch.load(args.saved_file_path)
	saved_image = torch.load(args.image_file_path)

	adv_images = saved_data['adv'].numpy()  # Adversarial images
	nat_images = saved_image['images']
	nat_images = nat_images[:-3].numpy()

	true_labels = saved_image['labels']  # True labels
	true_labels = true_labels[:-3].numpy()

	# Initialize a dictionary to store the first index of each unique label
	first_indices = {}

	# Iterate over the labels
	for idx, label in enumerate(true_labels):
		if label not in first_indices:
			first_indices[label] = idx

	print(first_indices)

def convert_img():

	path = "./Images/"
	if not os.path.exists(path):
		os.makedirs(path)

	#images = np.load('X.npy')

	saved_data = torch.load(args.saved_file_path)
	saved_image = torch.load(args.image_file_path)

	adv_images = saved_data['adv'].numpy()  # Adversarial images
	nat_images = saved_image['images']
	nat_images = nat_images[:-3].numpy()

	true_labels = saved_image['labels']  # True labels
	true_labels = true_labels[:-3].numpy()

	images = nat_images

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
	#combine_csv()
	find_icons()

if __name__ == '__main__':
	main()