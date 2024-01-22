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

def check_diff():

	saved_data = torch.load(args.saved_file_path)
	saved_image = torch.load(args.image_file_path)

	adv_images = saved_data['adv'].numpy()  # Adversarial images
	nat_images = saved_image['images']
	nat_images = nat_images[:-3].numpy()

	true_labels = saved_image['labels']  # True labels
	true_labels = true_labels[:-3].numpy()

	identical_count = 0
	for adv_img, nat_img in zip(adv_images, nat_images):
		if np.array_equal(adv_img, nat_img):
			identical_count += 1

	print(f'Number of identical images: {identical_count}')
	print(f'Total number of images: {len(adv_images)}')

def main():
	check_diff()

if __name__ == '__main__':
	main()