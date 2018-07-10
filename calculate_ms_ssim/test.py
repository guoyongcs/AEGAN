import os
import random

files = os.listdir('/home/chenqi/Desktop/GANs/guoyong/rsgan-ours-test/generate_image_embgan_two_stage/flowers/output_flowers')
random.shuffle(files)

length = len(files)/2

for i in range(length):
	print(files[i*2], files[i*2+1])








