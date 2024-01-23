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
from torchvision.models import resnet101# ResNet101_Weights
import torchvision.models as models
#from contrastive import CPCA
import numpy as np

#from GTSRB import GTSRB_Test
#from feature_extractor import FeatureExtractor

import cv2
from torch.autograd import Function

from pytorch_grad_cam import GradCAM, HiResCAM, EigenGradCAM, ScoreCAM, GradCAMPlusPlus, XGradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


parser = argparse.ArgumentParser(description='Data Preparation for HAM10000')

parser.add_argument('--model-num', type=int, default=0, help='which model checkpoint to use')
#parser.add_argument('--model-path',
					#default='./checkpoints/model_gtsrb_rn_adv6.pt',
					#help='model for white-box attack evaluation') #nat, adv1, adv6
parser.add_argument('--test-batch-size', type=int, default=50, metavar='N',
					help='input batch size for testing (default: 200)')
parser.add_argument('--no-cuda', action='store_true', default=False,
					help='disables CUDA training')
parser.add_argument('--saved_file_path', type=str, default='./checkpoints/pixel_vgg_1103_5000_96_0.2000_rand_standard.pth', help='Path to the saved adversarial images')
parser.add_argument('--image_file_path', type=str, default='./checkpoints/images_vgg_1103_96.pth', help='Path to the sampled images and labels')

args = parser.parse_args()

# settings
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

# set up data loader
#transform_test = transforms.Compose([
	#transforms.Resize((96, 96)),
	#transforms.ToTensor(),
#])

#testset = GTSRB_Test(
	#root_dir='/content/data/GTSRB/Final_Test/Images/',
	#transform=transform_test
#)

#testset = GTSRB_Test(
	#root_dir='/content/data/Images_2_ppm',
	#transform=transform_test
#)

input_size = 96

num_instance = 0

norm_mean = [0.7630392, 0.5456477, 0.57004845]
norm_std = [0.1409286, 0.15261266, 0.16997074]

# define the transformation of the val images.
val_transform = transforms.Compose([transforms.Resize((input_size,input_size)), transforms.ToTensor(),
									transforms.Normalize(norm_mean, norm_std)])

normalize_ = transforms.Normalize(norm_mean, norm_std)

dir_ = './grad-cam_/' + args.model_num + '-0'

#model-dataset
if not os.path.exists(dir_):
	os.makedirs(dir_)

#test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

def rep(backbone, model, device, test_loader, cam):
	model.eval()

	#feature list
	features = []
	#prediction list
	predictions = []
	#target list
	targets = []

	accu = 0
	total = 0

	counter = 0

	for batch_idx, (data, target) in enumerate(test_loader):
		data, target = data.to(device), target.to(device)
		X, y = Variable(data, requires_grad=True), Variable(target)  # Added requires_grad=True

		feat = backbone(normalize_(X))#.reshape(X.shape[0], 2048)
		feat = feat.view(feat.size(0), -1)

		#pass thru linear layer to obtain prediction result
		pred = model(normalize_(X))

		accu += (pred.data.max(1)[1] == y.data).float().sum()
		total += X.shape[0]

		targets.extend(y.data.cpu().detach().numpy())
		predictions.extend(pred.data.max(1)[1].cpu().detach().numpy())
		#push representation to the list
		features.extend(feat.cpu().detach().numpy())

		# New Grad-CAM code starts here
		#cam_targets = pred.data.max(1)[1].cpu().numpy()  # Get the class indices
		#grayscale_cam = cam(input_tensor=X, targets=None, aug_smooth=True, eigen_smooth=True)
		grayscale_cam = cam(input_tensor=normalize_(X), targets=None)

		for j in range(grayscale_cam.shape[0]):
			visualization = show_cam_on_image(data.cpu().numpy()[j].transpose(1, 2, 0), grayscale_cam[j, :])

			cv2.imwrite(f'{dir_}/gradcam_img{counter}.jpg', visualization)
			counter += 1

		#break
	
	#convert to numpy arrays
	targets = np.array(targets)
	predictions = np.array(predictions)
	features = np.array(features)

	print("Prediction Accuracy:" + str(accu/total))

	return features, predictions, targets
	#return predictions, targets

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

	#initialize model
	#model = resnet101()
	#model.fc = nn.Linear(2048, 43)

	model = models.vgg16()
	model.classifier[6] = nn.Linear(4096, 7)
	#model.load_state_dict(torch.load(args.model_path))
	#model = model.to(device)

	#print(model.state_dict().keys())

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

	backbone = model.features
	backbone = backbone.to(device)

	#backbone = FeatureExtractor(model)
	#backbone = backbone.to(device)

	#fc = model.fc

	#model_ = nn.Sequential(backbone, fc)

	# Initialize Grad-CAM
	#target_layers = [model.layer4[-1]]
	target_layers = [model.features[-1]]
	cam = GradCAM(model=model, target_layers=target_layers, use_cuda=use_cuda)

	features, predictions, targets = rep(backbone, model, device, test_loader_nat, cam)

	#convert to tabular data
	#path = "./tabu_data/"
	#if not os.path.exists(path):
		#os.makedirs(path)
	
	predictions = predictions.reshape(predictions.shape[0], 1)
	targets = targets.reshape(targets.shape[0], 1)

if __name__ == '__main__':
	main()