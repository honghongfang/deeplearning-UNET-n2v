#!/usr/bin/env python3

import os
import sys
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--baseDir", help="base directory in which your network will live", default='models')
parser.add_argument("--name", help="name of your network", default='N2V')
parser.add_argument("--dataPath", help="The path to your training data")
parser.add_argument("--fileName", help="name of your training data file", default="*.tif")
parser.add_argument("--validationFraction", help="Fraction of data you want to use for validation (percent)", default=5.0, type=float)
parser.add_argument("--dims", help="dimensions of your data, can include: X,Y,Z,C (channel), T (time)", default='YX')
parser.add_argument("--patchSizeXY", help="XY-size of your training patches", default=64, type=int)
parser.add_argument("--patchSizeZ", help="Z-size of your training patches", default=4, type=int)
parser.add_argument("--epochs", help="number of training epochs", default=100, type=int)
parser.add_argument("--stepsPerEpoch", help="number training steps per epoch", default=400, type=int)
parser.add_argument("--batchSize", help="size of your training batches", default=128, type=int)
#parser.add_argument("--netDepth", help="depth of your U-Net", default=2, type=int)
parser.add_argument("--netKernelSize", help="Size of conv. kernels in first layer", default=3, type=int)
parser.add_argument("--n2vPercPix", help="percentage of pixels to manipulated by N2V", default=1.6, type=float)
parser.add_argument("--learningRate", help="initial learning rate", default=0.0004, type=float)
parser.add_argument("--unet_n_first", help="number of feature channels in the first u-net layer", default=32, type=int)
#parser.add_argument("--noAugment",  action='store_true', help="do not rotate and flip training patches")

if len(sys.argv)==1:
    parser.print_help(sys.stderr)
    sys.exit(1)

args = parser.parse_args()
print(args)

from n2v.models import N2VConfig, N2V
print('everything imported')
import numpy as np
from csbdeep.utils import plot_history
from n2v.utils.n2v_utils import manipulate_val_data
from n2v.internals.N2V_DataGenerator import N2V_DataGenerator
from matplotlib import pyplot as plt
import urllib
import os
import zipfile
import random

from tifffile import imread
from tifffile import imwrite


import glob
print('everything imported')


print("args",str(args.name))

#print('augment',(not args.noAugment))



####################################################
#           PREPARE TRAINING DATA
####################################################


datagen = N2V_DataGenerator()
#imgs = datagen.load_imgs_from_directory(directory = args.dataPath, dims=args.dims, filter=args.fileName)
imgs = datagen.load_imgs_from_directory(directory=args.dataPath, dims='ZYXC', filter=args.fileName)

print("imgs.shape",imgs[0].shape)


# Here we extract patches for training and validation.
#Load one randomly chosen training target file

random_choice=random.choice(os.listdir(args.dataPath))
x = imread(args.dataPath+"/"+random_choice)

# Here we check that the input images are stacks
if len(x.shape) == 2:
  print("Image dimensions (y,x)",x.shape)

if not len(x.shape) == 2:
  print(bcolors.WARNING + "Your images appear to have the wrong dimensions. Image dimension",x.shape)



#Find image XY dimension
Image_Y = x.shape[0]
Image_X = x.shape[1]

#Hyperparameters failsafes

# Here we check that args.patchSizeXY is smaller than the smallest xy 
# dimension of the image 
if args.patchSizeXY > min(Image_Y, Image_X):
  args.patchSizeXY = min(Image_Y, Image_X)
  print (bcolors.WARNING + " Your chosen args.patchSizeXY is bigger than the xy dimension of your image; therefore the args.patchSizeXY chosen is now:",args.patchSizeXY)

# Here we check that args.patchSizeXY is divisible by 8
if not args.patchSizeXY % 8 == 0:
    args.patchSizeXY = ((int(args.patchSizeXY / 8)-1) * 8)
    print (bcolors.WARNING + " Your chosen args.patchSizeXY is not divisible by 8; therefore the args.patchSizeXY chosen is now:",args.patchSizeXY)

#patches = datagen.generate_patches_from_list(imgs[:1], shape=pshape, augment=True)
#Xdata = datagen.generate_patches_from_list(imgs, shape=(args.patchSizeXY, args.patchSizeXY), augment=True)
Xdata = datagen.generate_patches_from_list(imgs, shape=(args.patchSizeZ, args.patchSizeXY, args.patchSizeXY, num_channels), augment=True)

shape_of_Xdata = Xdata.shape


# create a threshold (10 % patches for the validation)
threshold = int(shape_of_Xdata[0]*(args.validationFraction/100))
# split the patches into training patches and validation patches
X = Xdata[threshold:]
X_val = Xdata[:threshold]
print(Xdata.shape[0],"patches created.")
print(threshold,"patch images for validation (",args.validationFraction,"%).")
print(Xdata.shape[0]-threshold,"patch images for training.")




# create a Config object
config = N2VConfig(X, unet_kern_size=args.netKernelSize, 
                   train_steps_per_epoch=int(args.stepsPerEpoch), train_epochs=int(args.epochs), 
                   train_loss='mse', batch_norm=True, train_batch_size=args.batchSize, n2v_perc_pix=args.n2vPercPix, 
                   n2v_manipulator='uniform_withCP', n2v_neighborhood_radius=5, train_learning_rate =args.learningRate)
# Let's look at the parameters stored in the config-object.
vars(config)


input_shape = (args.patchSizeZ, args.patchSizeXY, args.patchSizeXY, num_channels)
config = N2VConfig(input_shape, unet_kern_size=args.netKernelSize, 
                   n2v_perc_pix=args.n2vPercPix, 
                   train_steps_per_epoch=args.stepsPerEpoch, train_epochs=args.epochs, batch_norm=True, 
                   learning_rate=args.learningRate, train_batch_size=args.batchSize, 
                   n2v_manipulator='uniform_withCP', unet_n_first=args.unet_n_first)
                
# a name used to identify the model
model_name = args.name
# the base directory in which our model will live
model_path = args.baseDir        
# create network model.
model = N2V(config=config, name=model_name, basedir=model_path)





####################################################
#           Train Network
####################################################

history = model.train(X, X_val)
print("Training done.")
