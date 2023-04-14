# --------------------------------------------
# MNIST Neural Network Challenge.
# Credit : Nicolas Ronette's video
# https://www.youtube.com/watch?v=mozBidd58VQ
# --------------------------------------------

# --------------------------------------------
# Importing the dependences.
# Torch
from torch import nn   # our Neural Network Class
from torch.optim import Adam  # our Optimizer
from torch.utils.data import DataLoader  # our Data Loader
# Torch Vision
from torchvision import datasets # to download our MNIST dataset
from torchvision.transforms import ToTensor # to transfrom our images into tensors
#  Needed to be able to Load / Use the 'model.pt' file
import torch
from PIL import Image







