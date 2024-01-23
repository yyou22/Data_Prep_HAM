from __future__ import print_function
from sklearn.manifold import TSNE
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.models import resnet101, ResNet101_Weights
import torchvision.models as models
#from contrastive import CPCA
import numpy as np
import pandas as pd
from PIL import Image

from torch.utils.data import Dataset

parser = argparse.ArgumentParser(description='Data Preparation for Traffic Sign Project')

parser.add_argument('--model-num', type=int, default=1, help='which model checkpoint to use')
parser.add_argument('--test-batch-size', type=int, default=50, metavar='N',
					help='input batch size for testing (default: 200)')
parser.add_argument('--no-cuda', action='store_true', default=False,
					help='disables CUDA training')
parser.add_argument('--saved_file_path', type=str, default='./checkpoints/pixel_vgg_1103_5000_96_0.2000_rand_beta.5.pth', help='Path to the saved adversarial images')
parser.add_argument('--image_file_path', type=str, default='./checkpoints/images_vgg_1103_96.pth', help='Path to the sampled images and labels')

args = parser.parse_args()

# settings
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

input_size = 96

num_instance = 0

norm_mean = [0.7630392, 0.5456477, 0.57004845]
norm_std = [0.1409286, 0.15261266, 0.16997074]

# define the transformation of the val images.
val_transform = transforms.Compose([transforms.Resize((input_size,input_size)), transforms.ToTensor(),
									transforms.Normalize(norm_mean, norm_std)])

normalize_ = transforms.Normalize(norm_mean, norm_std)


def rep(backbone, model, device, test_loader_nat, test_loader_adv):
	model.eval()

	#feature list
	features = []
	#prediction list
	predictions = []
	#target list
	targets = []
	#show nat or adv
	type_ = []
	match_idx = []

	nat_accu = 0
	nat_total = 0

	for data,target in test_loader_nat:

		data, target = data.to(device), target.to(device)
		X, y = Variable(data), Variable(target)

		#feat = model[0](normalize_(X)).reshape(X.shape[0], 4096)

		feat = backbone(normalize_(X))
		feat = feat.view(feat.size(0), -1)  # Using view for flattening
		#print(feat.shape)

		#pass thru linear layer to obtain prediction result
		pred = model(normalize_(X))

		nat_accu += (pred.data.max(1)[1] == y.data).float().sum()
		nat_total += X.shape[0]

		targets.extend(y.data.cpu().detach().numpy())
		predictions.extend(pred.data.max(1)[1].cpu().detach().numpy())
		#push representation to the list
		features.extend(feat.cpu().detach().numpy())

	type_.extend([0] * nat_total)

	for i in range(nat_total):
		match_idx.append(i)
	match_idx = match_idx[:nat_total]
	match_idx.extend(match_idx)

	print(match_idx)

	print("Natural Prediction Accuracy:" + str(nat_accu/nat_total))

	adv_accu = 0
	adv_total = 0

	for data,target in test_loader_adv:
		data, target = data.to(device), target.to(device)
		X, y = Variable(data), Variable(target)

		#feat = model[0](normalize_(X)).reshape(X.shape[0], 4096)

		feat = backbone(normalize_(X))
		feat = feat.view(feat.size(0), -1)  # Using view for flattening
		#print(feat.shape)

		#pass thru linear layer to obtain prediction result
		pred = model(normalize_(X))

		adv_accu += (pred.data.max(1)[1] == y.data).float().sum()
		adv_total += X.shape[0]

		targets.extend(y.data.cpu().detach().numpy())
		predictions.extend(pred.data.max(1)[1].cpu().detach().numpy())
		#push representation to the list
		features.extend(feat.cpu().detach().numpy())

	type_.extend([1] * adv_total)

	print("Adversarial Prediction Accuracy:" + str(adv_accu/adv_total))
	
	#convert to numpy arrays
	targets = np.array(targets)
	predictions = np.array(predictions)
	features = np.array(features)
	type_ = np.array(type_)
	match_idx = np.array(match_idx)

	print('predictions.shape', predictions.shape)

	return features, predictions, targets, type_, match_idx

def dimen_reduc(features):
	
	feature_t = TSNE_(features)

	tx, ty = feature_t[:, 0].reshape(num_instance *2, 1), feature_t[:, 1].reshape(num_instance *2, 1)
	tx = (tx-np.min(tx)) / (np.max(tx) - np.min(tx))
	ty = (ty-np.min(ty)) / (np.max(ty) - np.min(ty))

	return tx, ty

def TSNE_(data):

	tsne = TSNE(n_components=2)
	data = tsne.fit_transform(data)

	print(data.shape)

	return data

def main():

	saved_data = torch.load(args.saved_file_path, map_location=device)
	saved_image = torch.load(args.image_file_path, map_location=device)

	adv_images = saved_data['adv']#.detach().cpu().numpy()  # Adversarial images
	nat_images = saved_image['images']
	nat_images = nat_images[:-3]#.detach().cpu().numpy()

	true_labels = saved_image['labels']  # True labels
	true_labels = true_labels[:-3]#.detach().cpu().numpy()

	label_4_indexes = (true_labels == 4).nonzero(as_tuple=True)[0]

	# Select the first 400 instances where label is 4
	excluded_indexes = label_4_indexes[:400]
	# Create a mask for all indices
	mask = torch.ones(len(true_labels), dtype=torch.bool)
	# Set the mask to false for excluded indexes
	mask[excluded_indexes] = False

	# Extract the corresponding images
	nat_images = nat_images[mask]
	adv_images = adv_images[mask]
	true_labels = true_labels[mask]

	print("Shape of adv_images:", adv_images.shape)
	print("Shape of nat_images:", nat_images.shape)
	print("Shape of true_labels:", true_labels.shape)

	global num_instance
	num_instance = true_labels.shape[0]

	model = models.vgg16()
	model.classifier[6] = nn.Linear(4096, 7)

	testset_adv = torch.utils.data.TensorDataset(adv_images.detach().cpu(), true_labels.detach().cpu())
	test_loader_adv = torch.utils.data.DataLoader(testset_adv, batch_size=args.test_batch_size, shuffle=False, **kwargs)

	testset_nat = torch.utils.data.TensorDataset(nat_images.detach().cpu(), true_labels.detach().cpu())
	test_loader_nat = torch.utils.data.DataLoader(testset_nat, batch_size=args.test_batch_size, shuffle=False, **kwargs)

	if args.model_num == 0:
		model_path = './checkpoints/standard.pt'
	elif args.model_num == 1:
		model_path = './checkpoints/beta.5.pt'
	elif args.model_num == 2:
		model_path = './checkpoints/beta4.pt'

	model.load_state_dict(torch.load(model_path))
	model = model.to(device)

	#print(model.state_dict().keys())

	#backbone = FeatureExtractor(model)
	#backbone = backbone.to(device)

	#for vgg models, you can directly extract the backbone by using .features
	backbone = model.features
	backbone = backbone.to(device)

	features, predictions, targets, type_, match_idx = rep(backbone, model, device, test_loader_nat, test_loader_adv)

	indices = np.where(targets == 0)[0]
	print(indices)
	print(len(indices))

	# Get the subset of data instances with the target label
	filtered_data = features[indices]

	print(features.shape)

	tx, ty = dimen_reduc(features)
	#tx, ty = dimen_reduc_cpca(filtered_data)

	#convert to tabular data
	path = "./tabu_data/"
	if not os.path.exists(path):
		os.makedirs(path)
	
	predictions = predictions.reshape(predictions.shape[0], 1)
	targets = targets.reshape(targets.shape[0], 1)
	type_ = type_.reshape(type_.shape[0], 1)
	match_idx = match_idx.reshape(match_idx.shape[0], 1)

	#only for one class
	#predictions = predictions[indices]
	#targets = targets[indices]
	#type_ = type_[indices]

	print('tx.shape', tx.shape)
	print('ty.shape', ty.shape)

	result = np.concatenate((tx, ty, predictions, targets, type_, match_idx), axis=1)
	type_ = ['%.5f'] * 2 + ['%d'] * 4
	np.savetxt(path + "data_" + str(args.model_num) + "_all.csv", result, header="xpos,ypos,pred,target,type,match_idx", comments='', delimiter=',', fmt=type_)

if __name__ == '__main__':
	main()