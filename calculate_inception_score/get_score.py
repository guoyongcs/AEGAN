
import inception_score
import argparse
import os
import numpy as np
from PIL import Image
import random


def get_images(path, imageSize):
	# get the images' name
	images_files = os.listdir(path)

	# shuffle the order of files' name
	random.shuffle(images_files)

	# transform the images to the numpy array
	images_list = []
	for file in images_files:
		img = Image.open(os.path.join(path, file))
		img = img.resize((imageSize, imageSize), Image.ANTIALIAS)
		img = np.array(img)
		images_list.append(img)

	return images_list


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', required=True, help='name of dataset')
	parser.add_argument('--dataroot', required=True, help='path to images')
	parser.add_argument('--log_name', required=True, help='name of log file')
	parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
	opt = parser.parse_args()
	print(opt)

	# get the list of images as inputs of calculating inception score
	images_list = get_images(opt.dataroot, opt.imageSize)

	# calculate the inception score
	mean_score, std_score = inception_score.get_inception_score(images=images_list, log_file=opt.log_name)
	print("mean score : {}".format(mean_score))
	print("std score : {}".format(std_score))


if __name__ == '__main__':
	# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
	# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
	main()


