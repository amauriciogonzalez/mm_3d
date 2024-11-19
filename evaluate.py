# Importing libraries
from torch.utils.data import DataLoader
from torchvision import transforms
from mm_shap_e import ObjaverseDataset, create_train_test_split, evaluate_objaverse_instance, MMShapE, evaluate_objaverse_instance_DF

import argparse
import importlib
import logging
import os
import sys
import time
import traceback

import torch.cuda

import matplotlib.pyplot as plt




print("initializing dataset")
image_dir = "/lustre/fs1/home/cap6411.student4/Project/mvs_objaverse/Objaverse_Dataset/rendered_images/"
caption_csv = "/lustre/fs1/home/cap6411.student4/Project/mvs_objaverse/objaverse_csv.csv"
latent_code_dir = "/lustre/fs1/home/cap6411.student3/final_project/Cap3D/Cap3D_latentcodes/"
transform = transforms.ToTensor()

dataset = ObjaverseDataset(
  image_dir=image_dir,
  caption_csv=caption_csv,
  latent_code_dir=latent_code_dir,
  transform=transform
)

# Leave num_samples=0 to load all samples!
train_loader, test_loader = create_train_test_split(dataset, test_size=0.05, batch_size=1, num_samples=30)

print(len(test_loader))
print("initialized loaders")

print("Initialiing model")
model = MMShapE().to("cuda")
print("Evaluting model")
if __name__ == '__main__':
  print(evaluate_objaverse_instance_DF(model, test_loader))
